#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "PhysicsTools/IsolationAlgos/interface/CITKIsolationConeDefinitionBase.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <unordered_map>

namespace reco {
  typedef edm::Ptr<reco::GsfElectron> GsfElectronPtr;
}

namespace pat {
  typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}

class ElectronMiniIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  ElectronMiniIsolationWithConeVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _vetoConeSize2EB(std::pow(c.getParameter<double>("VetoConeSizeBarrel"),2.0)),
    _vetoConeSize2EE(std::pow(c.getParameter<double>("VetoConeSizeEndcaps"),2.0)),
    _minConeSize2(std::pow(c.getParameter<double>("MinConeSize"),2.0)),
    _ktScale(c.getParameter<double>("ktScale")),
    _actConeSize2(c.existsAs<double>("ActivityConeSize") ? std::pow(c.getParameter<double>("ActivityConeSize"),2.) : -1.),
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")),
    _isolateAgainst(c.getParameter<std::string>("isolateAgainst")) {
    char buf[50];
    if(_actConeSize2 <= 0.){
      sprintf(buf,"BarVeto%.2f-EndVeto%.2f-kt%.2f-Min%.2f",
	      std::sqrt(_vetoConeSize2EB),
	      std::sqrt(_vetoConeSize2EE),
	      _ktScale,
	      std::sqrt(_minConeSize2));
    }else{
      sprintf(buf,"BarVeto%.2f-EndVeto%.2f-kt%.2f-Min%.2f-Act%.2f",
	      std::sqrt(_vetoConeSize2EB),
	      std::sqrt(_vetoConeSize2EE),
	      _ktScale,
	      std::sqrt(_minConeSize2),
	      std::sqrt(_actConeSize2));
    }
    _additionalCode = std::string(buf);
    auto decimal = _additionalCode.find('.');
    while( decimal != std::string::npos ) {
      _additionalCode.erase(decimal,1);
      decimal = _additionalCode.find('.');
    }
  }
  ElectronMiniIsolationWithConeVeto(const ElectronMiniIsolationWithConeVeto&) = delete;
  ElectronMiniIsolationWithConeVeto& operator=(const ElectronMiniIsolationWithConeVeto&) =delete;

  void setConsumes(edm::ConsumesCollector) {}

  bool isInIsolationCone(const reco::CandidatePtr& physob,
                         const reco::CandidatePtr& other) const override final;

  //! Destructor
  virtual ~ElectronMiniIsolationWithConeVeto(){};

private:
  const float _vetoConeSize2EB, _vetoConeSize2EE;
  const float _minConeSize2, _ktScale;
  const float _actConeSize2;
  const std::vector<unsigned> _miniAODVertexCodes;
  const std::string _isolateAgainst;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
                  ElectronMiniIsolationWithConeVeto,
                  "ElectronMiniIsolationWithConeVeto");

bool ElectronMiniIsolationWithConeVeto::
isInIsolationCone(const reco::CandidatePtr& physob,
                  const reco::CandidatePtr& iso_obj  ) const {
  reco::GsfElectronPtr eleref(physob);
  pat::PackedCandidatePtr aspacked(iso_obj);
  reco::PFCandidatePtr aspf(iso_obj);
  const reco::CaloClusterPtr& seed = eleref->superCluster()->seed();
  bool isEB = ( seed->seed().subdetId() == EcalBarrel );
  const float deltar2 = reco::deltaR2(*physob,*iso_obj);
  const float vetoConeSize2 = ( isEB ? _vetoConeSize2EB : _vetoConeSize2EE );
  const float coneSize2 = std::max(_minConeSize2,
				   std::min(static_cast<float>(_coneSize2),
					    std::pow(static_cast<float>(_ktScale/eleref->pt()),2.f)));
  bool result = true;
  if( aspacked.isNonnull() && aspacked.get() ) {
    if( aspacked->charge() != 0 ) {
      bool is_vertex_allowed = false;
      for( const unsigned vtxtype : _miniAODVertexCodes ) {
        if( vtxtype == aspacked->fromPV() ) {
          is_vertex_allowed = true;
          break;
        }
      }
      result *= ( is_vertex_allowed );
    }
    if(_actConeSize2 <= 0.){
      result *= deltar2 > vetoConeSize2 && deltar2 < coneSize2 ;
    }else{
      result *= deltar2 > vetoConeSize2 && deltar2 >= coneSize2 && deltar2 < _actConeSize2;
    }
  } else if ( aspf.isNonnull() && aspf.get() ) {
    if(_actConeSize2 <= 0.){
      result *= deltar2 > vetoConeSize2 && deltar2 < coneSize2;
    }else{
      result *= deltar2 > vetoConeSize2 && deltar2 >= coneSize2 && deltar2 < _actConeSize2;
    }
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
