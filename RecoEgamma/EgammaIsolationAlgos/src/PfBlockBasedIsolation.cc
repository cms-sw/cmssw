#include "RecoEgamma/EgammaIsolationAlgos/interface/PfBlockBasedIsolation.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"



#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"


//--------------------------------------------------------------------------------------------------

PfBlockBasedIsolation::PfBlockBasedIsolation() {
  // Default Constructor.
}




//--------------------------------------------------------------------------------------------------
PfBlockBasedIsolation::~PfBlockBasedIsolation()
{

}


void  PfBlockBasedIsolation::setup ( const edm::ParameterSet& conf ) {

  coneSize_             = conf.getParameter<double>("coneSize");  

}


std::vector<reco::PFCandidateRef>  PfBlockBasedIsolation::calculate(math::XYZTLorentzVectorD p4, const reco::PFCandidateRef pfEGCand, const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle) {
  
  std::vector<reco::PFCandidateRef> myVec;
  
  math::XYZVector candidateMomentum(p4.px(),p4.py(),p4.pz());
  math::XYZVector candidateDirection=candidateMomentum.Unit();
  //  std::cout << " PfBlockBasedIsolation::calculate photon momentum direction " << candidateDirection << " eta " << candidateDirection.Eta() << " phi " << candidateDirection.Phi() << " pfEGCand SC energy " <<   pfEGCand->superClusterRef()->energy() << std::endl; 
  const reco::PFCandidate::ElementsInBlocks& theElementsInpfEGcand = (*pfEGCand).elementsInBlocks();
  reco::PFCandidate::ElementsInBlocks::const_iterator ieg = theElementsInpfEGcand.begin();
  const reco::PFBlockRef egblock = ieg->first;


  unsigned nObj = pfCandidateHandle->size();
  for(unsigned int lCand=0; lCand < nObj; lCand++) {

    reco::PFCandidateRef pfCandRef(reco::PFCandidateRef(pfCandidateHandle,lCand));

    float dR = deltaR(candidateDirection.Eta(), candidateDirection.Phi(),  pfCandRef->eta(),   pfCandRef->phi());   
    //    std::cout << " PfBlockBasedIsolation::calculate candidate type " << pfCandRef->particleId() << " dR  " << dR << std::endl;       
    if ( dR> coneSize_ ) continue;

    const reco::PFCandidate::ElementsInBlocks& theElementsInPFcand = pfCandRef->elementsInBlocks();

    bool elementFound=false;
    for (reco::PFCandidate::ElementsInBlocks::const_iterator ipf = theElementsInPFcand.begin(); ipf<theElementsInPFcand.end(); ++ipf) {
 
     if ( ipf->first == egblock && !elementFound ) {
       //	std::cout << " this is the same block as the unbiassed cand  and this is an element " << ipf->second<< std::endl;
	  for (ieg = theElementsInpfEGcand.begin(); ieg<theElementsInpfEGcand.end(); ++ieg) {
	    if ( ipf->second == ieg->second && !elementFound  ) {
	      elementFound=true;
	      myVec.push_back(pfCandRef);    
	      //     std::cout << " EG candidate has at leaast one same element " <<  ieg->second << " elementFound " << elementFound <<  std::endl;  
	    }
	  }
	
	
	
      }
    }

    //  std::cout << " PfBlockBasedIsolation myVec size " << myVec.size() << std::endl;
    //for ( std::vector<reco::PFCandidateRef>::const_iterator iPair=myVec.begin(); iPair<myVec.end(); iPair++) {
    // float dR = deltaR(candidateDirection.Eta(), candidateDirection.Phi(),  (*iPair)->eta(),   (*iPair)->phi());   
    // std::cout << " Pair " << (*iPair)->particleId() <<  " dR local " << dR << " pt " << (*iPair)->pt() <<  " eta " << (*iPair)->eta() << " phi " << (*iPair)->phi() << std::endl; 
    // }

  }
  


  return myVec;



 }




