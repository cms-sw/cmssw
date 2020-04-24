#include "RecoEgamma/EgammaIsolationAlgos/interface/PFBlockBasedIsolation.h"
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
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"

//--------------------------------------------------------------------------------------------------

PFBlockBasedIsolation::PFBlockBasedIsolation() {
  // Default Constructor.
}




//--------------------------------------------------------------------------------------------------
PFBlockBasedIsolation::~PFBlockBasedIsolation()
{

}


void  PFBlockBasedIsolation::setup ( const edm::ParameterSet& conf ) {

  coneSize_             = conf.getParameter<double>("coneSize");  

}


std::vector<reco::PFCandidateRef>  PFBlockBasedIsolation::calculate(math::XYZTLorentzVectorD p4, const reco::PFCandidateRef pfEGCand, const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle) {
  
  std::vector<reco::PFCandidateRef> myVec;
  
  math::XYZVector candidateMomentum(p4.px(),p4.py(),p4.pz());
  math::XYZVector candidateDirection=candidateMomentum.Unit();

  const reco::PFCandidate::ElementsInBlocks& theElementsInpfEGcand = (*pfEGCand).elementsInBlocks();
  reco::PFCandidate::ElementsInBlocks::const_iterator ieg = theElementsInpfEGcand.begin();
  const reco::PFBlockRef egblock = ieg->first;


  unsigned nObj = pfCandidateHandle->size();
  for(unsigned int lCand=0; lCand < nObj; lCand++) {

    reco::PFCandidateRef pfCandRef(reco::PFCandidateRef(pfCandidateHandle,lCand));

    float dR = 0.0;
    if( coneSize_ < 10.0 ) {
      dR = deltaR(candidateDirection.Eta(), candidateDirection.Phi(),  pfCandRef->eta(),   pfCandRef->phi());         
      if ( dR> coneSize_ ) continue;
    }

    const reco::PFCandidate::ElementsInBlocks& theElementsInPFcand = pfCandRef->elementsInBlocks();

    bool elementFound=false;
    for (reco::PFCandidate::ElementsInBlocks::const_iterator ipf = theElementsInPFcand.begin(); ipf<theElementsInPFcand.end(); ++ipf) {
 
     if ( ipf->first == egblock && !elementFound ) {

	  for (ieg = theElementsInpfEGcand.begin(); ieg<theElementsInpfEGcand.end(); ++ieg) {
	    if ( ipf->second == ieg->second && !elementFound  ) {
	      if(elementPassesCleaning(pfCandRef,pfEGCand)){
		myVec.push_back(pfCandRef);    
		elementFound=true;
	      }
	    }
	  }
	
	
	
      }
    }

    

  }
  


  return myVec;



 }


bool PFBlockBasedIsolation::elementPassesCleaning(const reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand)
{
  if(pfCand->particleId()==reco::PFCandidate::h) return passesCleaningChargedHadron(pfCand,pfEGCand);
  else if(pfCand->particleId()==reco::PFCandidate::h0) return passesCleaningNeutralHadron(pfCand,pfEGCand);
  else if(pfCand->particleId()==reco::PFCandidate::gamma) return passesCleaningPhoton(pfCand,pfEGCand);
  else return true; //doesnt really matter here as if its not a photon,neutral or charged it wont be included in isolation
}

//currently the record of which candidates came from the charged hadron is acceptable, no further cleaning is needed
bool PFBlockBasedIsolation::passesCleaningChargedHadron(const reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand)
{
  return true;
}

//neutral hadrons are not part of the PF E/gamma reco, therefore they cant currently come from an electron/photon and so should be rejected
//but we still think there may be some useful info here and given we can easily
//fix this at AOD level, we will auto accept them for now and clean later
bool PFBlockBasedIsolation::passesCleaningNeutralHadron(const  reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand)
{
  return true;
}

//the highest et ECAL element of the photon must match to the electron superclusters or one of its sub clusters
bool PFBlockBasedIsolation::passesCleaningPhoton(const  reco::PFCandidateRef& pfCand,const reco::PFCandidateRef& pfEGCand)
{
  bool passesCleaning=false;
  const reco::PFBlockElementCluster* ecalClusWithMaxEt = getHighestEtECALCluster(*pfCand);
  if(ecalClusWithMaxEt){
    if(ecalClusWithMaxEt->superClusterRef().isNonnull() && 
       ecalClusWithMaxEt->superClusterRef()->seed()->seed()==pfEGCand->superClusterRef()->seed()->seed()){ //being sure to match, some concerned about different collections, shouldnt be but to be safe
      passesCleaning=true;
    }else{
      for(auto cluster : pfEGCand->superClusterRef()->clusters()){
	//the PF clusters there are in two different collections so cant reference match
	//but we can match on the seed id, no clusters can share a seed so if the seeds are 
	//equal, it must be the same cluster
	if(ecalClusWithMaxEt->clusterRef()->seed()==cluster->seed()) {
	  passesCleaning=true;
	}
      }//end of loop over clusters
    }
  }//end of null check for highest ecal cluster
  return passesCleaning;

}


const reco::PFBlockElementCluster* PFBlockBasedIsolation::getHighestEtECALCluster(const reco::PFCandidate& pfCand)
{
  float maxECALEt =-1;
  const reco::PFBlockElement* maxEtECALCluster=nullptr;
  const reco::PFCandidate::ElementsInBlocks& elementsInPFCand = pfCand.elementsInBlocks();
  for(auto& elemIndx : elementsInPFCand){
    const reco::PFBlockElement* elem = elemIndx.second<elemIndx.first->elements().size() ? &elemIndx.first->elements()[elemIndx.second] : nullptr;
    if(elem && elem->type()==reco::PFBlockElement::ECAL && elem->clusterRef()->pt()>maxECALEt){
      maxECALEt = elem->clusterRef()->pt();
      maxEtECALCluster = elem;
    }
    
  }
  return dynamic_cast<const reco::PFBlockElementCluster*>(maxEtECALCluster);
	
}
