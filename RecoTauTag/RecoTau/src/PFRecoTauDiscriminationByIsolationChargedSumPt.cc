#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByIsolationChargedSumPt.h"

void PFRecoTauDiscriminationByIsolationChargedSumPt::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
  
  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> myPFTauDiscriminants(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));
  
  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
    PFTau thePFTau=*thePFTauRef;

    //check for leading tracks
    if ((ManipulateTracks_insteadofChargedHadrCands_ && thePFTau.leadTrack().isNull() )|| 
       (!ManipulateTracks_insteadofChargedHadrCands_ && thePFTau.leadPFCand().isNull()) )
    {
       myPFTauDiscriminants->setValue(iPFTau, 0.);
       continue;
    }

    math::XYZVector isolationSumP3;
    if (ManipulateTracks_insteadofChargedHadrCands_)
    {
       const TrackRefVector& isolationTracks = thePFTau.isolationTracks();
       for(size_t iTrack = 0; iTrack < isolationTracks.size(); ++iTrack)
       {
          float trackPt = isolationTracks[iTrack]->pt(); 
          if(trackPt > minPtForInclusion_) {
             isolationSumP3 += isolationTracks[iTrack]->momentum();
          }
       }
    } 
    else 
    { //use pf candidates instead
       const PFCandidateRefVector& pfIsoChargedCands = thePFTau.isolationPFChargedHadrCands();
       for(size_t iIsoCand = 0; iIsoCand < pfIsoChargedCands.size(); ++iIsoCand)
       {
          float pfChargedPt = pfIsoChargedCands[iIsoCand]->pt(); 
          if(pfChargedPt> minPtForInclusion_) {
             isolationSumP3 += pfIsoChargedCands[iIsoCand]->momentum();
          }
       }
    }

    float decision = 1.;
    if (isolationSumP3.perp2() > maxChargedSumPt_*maxChargedSumPt_)
       decision = 0.;

    myPFTauDiscriminants->setValue(iPFTau, decision);

  }    
  
  iEvent.put(myPFTauDiscriminants);
  
}


