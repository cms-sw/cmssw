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

class ElectronPFIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  ElectronPFIsolationWithConeVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _vetoConeSize2EB(std::pow(c.getParameter<double>("VetoConeSizeBarrel"),2.0)),
    _vetoConeSize2EE(std::pow(c.getParameter<double>("VetoConeSizeEndcaps"),2.0)),
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")),
    _isolateAgainst(c.getParameter<std::string>("isolateAgainst")) {
    char buf[50];
    sprintf(buf,"BarVeto%.2f-EndVeto%.2f",
	    std::sqrt(_vetoConeSize2EB),
	    std::sqrt(_vetoConeSize2EE));
    _additionalCode = std::string(buf);
    auto decimal = _additionalCode.find('.');
    while( decimal != std::string::npos ) {
      _additionalCode.erase(decimal,1);
      decimal = _additionalCode.find('.');
    }    
  }
  ElectronPFIsolationWithConeVeto(const ElectronPFIsolationWithConeVeto&) = delete;
  ElectronPFIsolationWithConeVeto& operator=(const ElectronPFIsolationWithConeVeto&) =delete;
  
  void setConsumes(edm::ConsumesCollector) {}

  bool isInIsolationCone(const reco::CandidatePtr& physob,
			 const reco::CandidatePtr& other) const override final;
  
  //! Destructor
  virtual ~ElectronPFIsolationWithConeVeto(){};
  
private:    
  const float _vetoConeSize2EB, _vetoConeSize2EE;  
  const std::vector<unsigned> _miniAODVertexCodes;
  const std::string _isolateAgainst;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
		  ElectronPFIsolationWithConeVeto,
		  "ElectronPFIsolationWithConeVeto");

bool ElectronPFIsolationWithConeVeto::
isInIsolationCone(const reco::CandidatePtr& physob,
		  const reco::CandidatePtr& iso_obj  ) const {
  reco::GsfElectronPtr eleref(physob);
  pat::PackedCandidatePtr aspacked(iso_obj);
  reco::PFCandidatePtr aspf(iso_obj);
  const reco::CaloClusterPtr& seed = eleref->superCluster()->seed();
  bool isEB = ( seed->seed().subdetId() == EcalBarrel );
  const float deltar2 = reco::deltaR2(*physob,*iso_obj);  
  const float vetoConeSize2 = ( isEB ? _vetoConeSize2EB : _vetoConeSize2EE );
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
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else if ( aspf.isNonnull() && aspf.get() ) {    
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2;
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
