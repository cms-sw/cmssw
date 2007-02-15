#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducer.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/Common/interface/EDProduct.h" 
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"


TrackInfoProducer::TrackInfoProducer(const edm::ParameterSet& iConfig):
    conf_(iConfig),
    theAlgo_(iConfig),
    forwardPredictedStateTag_(iConfig.getParameter<std::string>( "forwardPredictedState" )),
    backwardPredictedStateTag_(iConfig.getParameter<std::string>( "backwardPredictedState" )),
    updatedStateTag_(iConfig.getParameter<std::string>( "updatedState" )),
    combinedStateTag_(iConfig.getParameter<std::string>( "combinedState" ))
{
  // forward predicted state 
  produces<reco::TrackInfoCollection>(forwardPredictedStateTag_);

  // backward predicted state 
  produces<reco::TrackInfoCollection>(backwardPredictedStateTag_);

  // backward predicted + forward predicted + measured hit position 
  produces<reco::TrackInfoCollection>(updatedStateTag_);

  // backward predicted + forward predicted
  produces<reco::TrackInfoCollection>(combinedStateTag_);

  produces<reco::TrackInfoTrackAssociationCollection>(forwardPredictedStateTag_);
  produces<reco::TrackInfoTrackAssociationCollection>(backwardPredictedStateTag_);
  produces<reco::TrackInfoTrackAssociationCollection>(updatedStateTag_);
  produces<reco::TrackInfoTrackAssociationCollection>(combinedStateTag_);
}


void TrackInfoProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
   //
  // create empty output collections
  //
  std::auto_ptr<reco::TrackInfoCollection>    outputFwdColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputBwdColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputUpdatedColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoCollection>    outputCombinedColl (new reco::TrackInfoCollection);

  std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationFwdColl (new reco::TrackInfoTrackAssociationCollection);
  std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationBwdColl (new reco::TrackInfoTrackAssociationCollection);
  std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationUpdatedColl (new reco::TrackInfoTrackAssociationCollection);
  std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationCombinedColl (new reco::TrackInfoTrackAssociationCollection);
  
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTracks");
  edm::InputTag RHTag = conf_.getParameter<edm::InputTag>("rechits");
    edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
    edm::Handle<reco::TrackCollection > trackCollection;
    edm::Handle<TrackingRecHitCollection> rechitscollection;
  try{  

    theEvent.getByLabel(TkTag,TrajectoryCollection);
    theEvent.getByLabel(TkTag,trackCollection);
    theEvent.getByLabel(RHTag,rechitscollection);
  } 
  catch (cms::Exception &e){ edm::LogInfo("TrackInfoProducer") << "cms::Exception caught!!!" << "\n" << e << "\n";}
    //
    //run the algorithm  
    //
    reco::TrackInfo outputFwd,outputBwd,outputUpdated, outputCombined;

    std::vector<Trajectory>::const_iterator traj_iterator;

    std::vector<unsigned int> trackid;
    for(traj_iterator=TrajectoryCollection->begin();traj_iterator!=TrajectoryCollection->end();++traj_iterator){//loop on trajectories
      //associate the trajectory to the track
      unsigned int idtk = 0;
      
      if(TrajectoryCollection->size()==1)trackid.push_back(idtk);
      else{
	TrajectoryStateOnSurface outertsos=0;
	TrajectoryStateOnSurface innertsos=0;
	
	if (traj_iterator->direction() == alongMomentum) {
	  outertsos = traj_iterator->lastMeasurement().updatedState();
	  innertsos = traj_iterator->firstMeasurement().updatedState();
	} else { 
	  outertsos = traj_iterator->firstMeasurement().updatedState();
	  innertsos = traj_iterator->lastMeasurement().updatedState();
	}
	GlobalPoint po = outertsos.globalParameters().position();
	GlobalVector vo = outertsos.globalParameters().momentum();
	
	GlobalPoint pi = innertsos.globalParameters().position();
	GlobalVector vi = innertsos.globalParameters().momentum();
	
	reco::TrackCollection::const_iterator tk_iterator;
	for(tk_iterator=trackCollection->begin();tk_iterator!=trackCollection->end();++tk_iterator){//loop on tracks
	  
	  GlobalPoint tkpo = GlobalPoint(tk_iterator->outerPosition().X(),tk_iterator->outerPosition().Y(),tk_iterator->outerPosition().Z());
	  GlobalVector tkvo = GlobalVector(tk_iterator->outerMomentum().X(),tk_iterator->outerMomentum().Y(),tk_iterator->outerMomentum().Z());
	  GlobalPoint tkpi = GlobalPoint(tk_iterator->innerPosition().X(),tk_iterator->innerPosition().Y(),tk_iterator->innerPosition().Z());
	  GlobalVector tkvi = GlobalVector(tk_iterator->innerMomentum().X(),tk_iterator->innerMomentum().Y(),tk_iterator->innerMomentum().Z());
	  
	  if(((vo-tkvo).mag()<1e-16)&&
	     ((po-tkpo).mag()<1e-16)&&
	     ((vi-tkvi).mag()<1e-16)&&
	     ((pi-tkpi).mag()<1e-16)){
	    trackid.push_back(idtk);
	  }
	  else {
	    LogDebug("TrackInfoProducer")<<"Track not associated trying an other track"; 
	  }
	  idtk++;
	}
      }
      theAlgo_.run(traj_iterator,&rechitscollection,
		   outputFwd,outputBwd,outputUpdated, outputCombined
		   );
      outputFwdColl->push_back(*(new reco::TrackInfo(outputFwd)));
      outputBwdColl->push_back(*(new reco::TrackInfo(outputBwd)));
      outputUpdatedColl->push_back(*(new reco::TrackInfo(outputUpdated)));
      outputCombinedColl->push_back(*(new reco::TrackInfo(outputCombined)));

    }
    //put everything in the event
    const edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfof = theEvent.put(outputFwdColl,forwardPredictedStateTag_ );
    const edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfob =   theEvent.put(outputBwdColl,backwardPredictedStateTag_);
    const edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfou =   theEvent.put(outputUpdatedColl,updatedStateTag_ );
    const edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfoc =   theEvent.put(outputCombinedColl,combinedStateTag_ );
    for(unsigned int i=0; i <trackid.size();++i){
      TIassociationFwdColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, trackid[i]),edm::Ref<reco::TrackInfoCollection>(rTrackInfof, i));
      TIassociationBwdColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, trackid[i]),edm::Ref<reco::TrackInfoCollection>(rTrackInfob, i));
      TIassociationUpdatedColl->insert( edm::Ref<reco::TrackCollection>(trackCollection,trackid[i] ),edm::Ref<reco::TrackInfoCollection>(rTrackInfou, i));
      TIassociationCombinedColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, trackid[i]),edm::Ref<reco::TrackInfoCollection>(rTrackInfoc, i)); 
    }
    theEvent.put(TIassociationFwdColl,forwardPredictedStateTag_ );
    theEvent.put(TIassociationBwdColl,backwardPredictedStateTag_);
    theEvent.put(TIassociationUpdatedColl,updatedStateTag_ );
    theEvent.put(TIassociationCombinedColl,combinedStateTag_ ); 
}




