#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

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
  typedef edm::Ptr<reco::Photon> recoPhotonPtr;
}

namespace pat {
  typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}

class PhotonPFIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  PhotonPFIsolationWithConeVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _vetoConeSize2EB(std::pow(c.getParameter<double>("VetoConeSizeBarrel"),2.0)),
    _vetoConeSize2EE(std::pow(c.getParameter<double>("VetoConeSizeEndcaps"),2.0)),
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")),
    _vertexIndex(c.getParameter<int> ("vertexIndex")),
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
  PhotonPFIsolationWithConeVeto(const PhotonPFIsolationWithConeVeto&) = delete;
  PhotonPFIsolationWithConeVeto& operator=(const PhotonPFIsolationWithConeVeto&) =delete;
  
  void setConsumes(edm::ConsumesCollector) {}

  bool isInIsolationCone(const reco::CandidatePtr& photon,
			 const reco::CandidatePtr& pfCandidate) const override final;
  
  //! Destructor
  virtual ~PhotonPFIsolationWithConeVeto(){};
  
private:    
  const float _vetoConeSize2EB, _vetoConeSize2EE;  
  const std::vector<unsigned> _miniAODVertexCodes;
  const unsigned _vertexIndex;
  const std::string _isolateAgainst;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
		  PhotonPFIsolationWithConeVeto,
		  "PhotonPFIsolationWithConeVeto");

bool PhotonPFIsolationWithConeVeto::
isInIsolationCone(const reco::CandidatePtr& photon,
		  const reco::CandidatePtr& pfCandidate  ) const {
  reco::recoPhotonPtr photonref(photon);
  pat::PackedCandidatePtr aspacked(pfCandidate);
  reco::PFCandidatePtr aspf(pfCandidate);
  const reco::CaloClusterPtr& seed = photonref->superCluster()->seed();
  bool isEB = ( seed->seed().subdetId() == EcalBarrel );
  const float deltar2 = reco::deltaR2(*photon,*pfCandidate);  
  const float vetoConeSize2 = ( isEB ? _vetoConeSize2EB : _vetoConeSize2EE );
  bool result = true;
  if( aspacked.isNonnull() && aspacked.get() ) {    
    if( aspacked->charge() != 0 ) {
      bool is_vertex_allowed = false;
      for( const unsigned vtxtype : _miniAODVertexCodes ) {
	if( vtxtype == aspacked->fromPV(_vertexIndex) ) {
	  is_vertex_allowed = true;
	  break;
	}
      }      
      result &= ( is_vertex_allowed );
    }
    result &= deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else if ( aspf.isNonnull() && aspf.get() ) {    
    result &= deltar2 > vetoConeSize2 && deltar2 < _coneSize2;
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
