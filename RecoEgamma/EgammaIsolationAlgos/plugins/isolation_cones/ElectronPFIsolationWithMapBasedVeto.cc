#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/ValueMap.h"
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
  typedef edm::Ptr<pat::Electron> patElectronPtr;
}
namespace{
	// This template function finds whether theCandidate is in thefootprint
	// // collection. It is templated to be able to handle both reco and pat
	// // photons (from AOD and miniAOD, respectively).
	//
	template <class T, class U>
	bool isInFootprint(const T& thefootprint, const U& theCandidate) 
	{
	    for ( auto itr = thefootprint.begin(); itr != thefootprint.end(); ++itr ) 
    	    { 
      	      if( itr->key() == theCandidate.key() ) return true;
   	    }
 	    return false;
	}
        //This function is needed because pfNoPileUpCandidates have changed keys, 
        ////and thus the solution is to use sourceCandidatePtr(0) 
        // // This function *shouldn't be used for packedCandidate*
       template <class T, class U>
        bool isInFootprintAlternative(const T& thefootprint, const U& theCandidate)
        {
            for ( auto itr = thefootprint.begin(); itr != thefootprint.end(); ++itr )
            {
              if( itr->key() == theCandidate->sourceCandidatePtr(0).key() ) return true;
            }
            return false;
        }

}
class ElectronPFIsolationWithMapBasedVeto : public citk::IsolationConeDefinitionBase {
public:
  ElectronPFIsolationWithMapBasedVeto(const edm::ParameterSet& c) :
    citk::IsolationConeDefinitionBase(c),
    _isolateAgainst(c.getParameter<std::string>("isolateAgainst")), //isolate against either h+, h0 or gamma
    _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")){} //quality flags to be used for association with the vertex configurable, the vertex can be chosen
        
  ElectronPFIsolationWithMapBasedVeto(const ElectronPFIsolationWithMapBasedVeto&) = delete;
  ElectronPFIsolationWithMapBasedVeto& operator=(const ElectronPFIsolationWithMapBasedVeto&) =delete;

  bool isInIsolationCone(const reco::CandidatePtr& photon,
			 const reco::CandidatePtr& other) const override final;
  
   
  // this object is needed for reco case
 edm::Handle< edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationMap;			 
 edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationToken_;
 
 virtual void getEventInfo(const edm::Event& iEvent)
  {
      iEvent.getByToken(particleBasedIsolationToken_, particleBasedIsolationMap); 
  };			 
  
  
  //As far as I understand now, the object particleBasedIsolationMap should be fixed, so we don't configure the name
  void setConsumes(edm::ConsumesCollector iC)
  {
      particleBasedIsolationToken_ = iC.mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef > > >(edm::InputTag("particleBasedIsolation", "gedGsfElectrons"));
  }

  //! Destructor
  virtual ~ElectronPFIsolationWithMapBasedVeto(){};
  
  
private:    
  const std::string _isolateAgainst, _vertexCollection;
  const std::vector<unsigned> _miniAODVertexCodes;

  
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory,
		  ElectronPFIsolationWithMapBasedVeto,
		  "ElectronPFIsolationWithMapBasedVeto");


//This function defines whether particular PFCandidate is inside of isolation cone of photon or not by checking deltaR and whether footprint removal for this candidate should be done. Additionally, for miniAOD charged hadrons from the PV are considered. *** For AOD this should be done by the corresponding sequence beforehand!!! ***
 
bool ElectronPFIsolationWithMapBasedVeto::isInIsolationCone(const reco::CandidatePtr& electron,  const reco::CandidatePtr& pfCandidate  ) const {
 
  //convert the electron and candidate objects to the corresponding pat or reco objects. What is used depends on what is user running on: miniAOD or AOD
  pat::patElectronPtr aspat_electronptr(electron);
  
  pat::PackedCandidatePtr aspacked(pfCandidate);
  reco::PFCandidatePtr aspf(pfCandidate);

  
  bool inFootprint = false;
  bool result = true;
  const float deltar2 = reco::deltaR2(*electron,*pfCandidate); //calculate deltaR2 distance between PFCandidate and photon
  
  // dealing here with patObjects: miniAOD case
  if ( aspacked.get() )    
  {
    inFootprint = isInFootprint(aspat_electronptr ->associatedPackedPFCandidates(),aspacked);
    
    //checking if the charged candidates come from the appropriate vertex
    if( aspacked->charge() != 0 ) 
    {
      bool is_vertex_allowed = false;
      for( const unsigned vtxtype : _miniAODVertexCodes ) 
      {
      	if( vtxtype == aspacked->fromPV() ) {
      	  is_vertex_allowed = true;
      	  break;
      	}
      }      
    
     result &= (is_vertex_allowed);
    }
     //return true if the candidate is inside the cone and not in the footprint
    result &= deltar2 < _coneSize2 && (!inFootprint);
   }
  
  // dealing here with recoObjects: AOD case
  else if ( aspf.get())
  {
      inFootprint = isInFootprintAlternative((*particleBasedIsolationMap)[electron], pfCandidate); 
      //return true if the candidate is inside the cone and not in the footprint
      result &= deltar2 < _coneSize2 && (!inFootprint);
  }
  
  // throw exception if it is not a patObject or recoObject
  else {
    throw cms::Exception("InvalidIsolationInput")
      << "The supplied candidate to be used as isolation "
      << "was neither a reco::Photon nor a pat::Photon!";
  }
  
  return result;
}
