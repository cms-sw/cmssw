#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositVetoFactory.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "PhysicsTools/IsolationAlgos/interface/CITKIsolationConeDefinitionBase.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


#include <unordered_map>

class ElectronPFIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  ElectronPFIsolationWithConeVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _vetoConeSize2EB(std::pow(c.getParameter<double>("VetoConeSizeBarrel"),2.0)),
    _vetoConeSize2EE(std::pow(c.getParameter<double>("VetoConeSizeEndcaps"),2.0)),
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")),
    _vtxTag(c.getParameter<edm::InputTag>("vertexSrc")) {
  }
  ElectronPFIsolationWithConeVeto(const ElectronPFIsolationWithConeVeto&) = delete;
  ElectronPFIsolationWithConeVeto& operator=(const ElectronPFIsolationWithConeVeto&) =delete;
  
  void setConsumes(edm::ConsumesCollector cc) override final {
    _vtxToken = cc.consumes<reco::VertexCollection>(_vtxTag);
  };
  
  void getEventInfo(const edm::Event& ev) override final {
    ev.getByToken(_vtxToken,_vtxHandle);
  }

  bool isInIsolationCone(const reco::CandidateBaseRef& physob,
			 const reco::CandidateBaseRef& other) const override final;
  
  //! Destructor
  virtual ~ElectronPFIsolationWithConeVeto(){};
  
private:    
  const float _vetoConeSize2EB, _vetoConeSize2EE;  
  const std::vector<unsigned> _miniAODVertexCodes;
  const edm::InputTag _vtxTag;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
  edm::Handle<reco::VertexCollection> _vtxHandle;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
		  ElectronPFIsolationWithConeVeto,
		  "ElectronPFIsolationWithConeVeto");

bool ElectronPFIsolationWithConeVeto::
isInIsolationCone(const reco::CandidateBaseRef& physob,
		  const reco::CandidateBaseRef& other  ) const {
  reco::GsfElectronRef eleref = physob.castTo<reco::GsfElectronRef>();
  pat::PackedCandidateRef aspacked;
  reco::PFCandidateRef aspf;   
  try {
    aspacked =  other.castTo<pat::PackedCandidateRef>();
  } catch ( cms::Exception& ) {
  }
  try {
    aspf = other.castTo<reco::PFCandidateRef>();
  } catch ( cms::Exception& ) {
  }
  const reco::CaloClusterPtr& seed = eleref->superCluster()->seed();
  bool isEB = ( seed->seed().subdetId() == EcalBarrel );
  const float deltar2 = reco::deltaR2(*physob,*other);  
  const float vetoConeSize2 = ( isEB ? _vetoConeSize2EB : _vetoConeSize2EE );
  bool result = true;
  if( aspacked.isNonnull() ) {    
    if( aspacked->charge() != 0 ) {
      bool is_vertex_allowed = false;
      for( const unsigned vtxtype : _miniAODVertexCodes ) {
	if( vtxtype == aspacked->fromPV() ) {
	  is_vertex_allowed = true;
	  break;
	}
      }
      result *= is_vertex_allowed;
    }
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else if ( aspf.isNonnull() ) {
    if( aspf->charge() != 0 && _vtxHandle->size() ) {      
      result *= aspf->vertex() == (*_vtxHandle)[0].position();
    }
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2;
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
