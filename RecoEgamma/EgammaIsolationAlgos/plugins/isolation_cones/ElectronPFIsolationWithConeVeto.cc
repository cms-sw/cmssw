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

  bool isInIsolationCone(const reco::CandidateBaseRef& physob,
			 const reco::CandidateBaseRef& other) const override final;
  
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
isInIsolationCone(const reco::CandidateBaseRef& physob,
		  const reco::CandidateBaseRef& other  ) const {
  reco::GsfElectronRef eleref = physob.castTo<reco::GsfElectronRef>();
  pat::PackedCandidateRef aspacked;
  reco::PFCandidateRef aspf;
  bool cast_failed = false;  
  try {
    aspf = other.castTo<reco::PFCandidateRef>();
  } catch ( cms::Exception& ) {
    cast_failed = true;
  }
  if( cast_failed ) { // user analysis code has to suffer a little...
    try {
      aspacked =  other.castTo<pat::PackedCandidateRef>();
    } catch ( cms::Exception& ) {    
    }
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
      result *= ( is_vertex_allowed );
    }
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2 ;
  } else if ( aspf.isNonnull() ) {    
    result *= deltar2 > vetoConeSize2 && deltar2 < _coneSize2;
  } else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
