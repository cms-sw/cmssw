#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "PhysicsTools/IsolationAlgos/interface/CITKIsolationConeDefinitionBase.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


#include <unordered_map>

using namespace reco;

namespace reco {
  typedef edm::Ptr<reco::Muon> muonPtr;
}

namespace pat {
  typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}

class MuonPFIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  MuonPFIsolationWithConeVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _vetoThreshold(std::pow(c.getParameter<double>("VetoThreshold"),2.0)),
    _vetoConeSize(std::pow(c.getParameter<double>("VetoConeSize"),2.0)),
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")),
    _isolateAgainst(c.getParameter<std::string>("isolateAgainst")) {
    char buf[50];
    sprintf(buf,"ThresholdVeto%.2f-ConeVeto%.2f",
	    std::sqrt(_vetoThreshold),
	    std::sqrt(_vetoConeSize));
    _additionalCode = std::string(buf);
    auto decimal = _additionalCode.find('.');
    while( decimal != std::string::npos ) {
      _additionalCode.erase(decimal,1);
      decimal = _additionalCode.find('.');
    }    
  }
  MuonPFIsolationWithConeVeto(const MuonPFIsolationWithConeVeto&) = delete;
  MuonPFIsolationWithConeVeto& operator=(const MuonPFIsolationWithConeVeto&) =delete;
  
  void setConsumes(edm::ConsumesCollector) {}

  bool isInIsolationCone(const reco::CandidatePtr& physob,
			 const reco::CandidatePtr& other) const override final;
  
  //! Destructor
  virtual ~MuonPFIsolationWithConeVeto(){};
  
private:    
  const double _vetoThreshold, _vetoConeSize;  
  const std::vector<unsigned> _miniAODVertexCodes;
  const std::string _isolateAgainst;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
		  MuonPFIsolationWithConeVeto,
		  "MuonPFIsolationWithConeVeto");

bool MuonPFIsolationWithConeVeto::
isInIsolationCone(const reco::CandidatePtr& physob,
		  const reco::CandidatePtr& iso_obj  ) const {
  reco::muonPtr muonref(physob);
  pat::PackedCandidatePtr aspacked(iso_obj);
  reco::PFCandidatePtr aspf(iso_obj);
  //bool isEB = ( abs(muonref->eta()) < 1.1 );
  const double deltar2 = reco::deltaR2(*physob,*iso_obj);  
  const double vetoConeSize2 =  _vetoConeSize;
  const double vetoThreshold =  _vetoThreshold;
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
    result *= aspacked->pt() >  vetoThreshold && deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else if ( aspf.isNonnull() && aspf.get() ) {    
    result *= aspf->pt() >  vetoThreshold && deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
