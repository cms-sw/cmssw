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
  try{  
    edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
    theEvent.getByLabel(TkTag,TrajectoryCollection);
    edm::Handle<reco::TrackCollection > trackCollection;
    theEvent.getByLabel(TkTag,TrajectoryCollection);
    edm::Handle<TrackingRecHitCollection> rechitscollection;
    theEvent.getByLabel(RHTag,rechitscollection);
    //
    //run the algorithm  
    //
    reco::TrackInfo *outputFwd=0,*outputBwd=0,*outputUpdated=0, *outputCombined=0;

    std::vector<Trajectory>::const_iterator traj_iterator;
    
    reco::TrackInfoRefProd rTrackInfof = theEvent.getRefBeforePut<reco::TrackInfoCollection>(forwardPredictedStateTag_);
    reco::TrackInfoRefProd rTrackInfob = theEvent.getRefBeforePut<reco::TrackInfoCollection>(backwardPredictedStateTag_);
    reco::TrackInfoRefProd rTrackInfou = theEvent.getRefBeforePut<reco::TrackInfoCollection>(updatedStateTag_);
    reco::TrackInfoRefProd rTrackInfoc = theEvent.getRefBeforePut<reco::TrackInfoCollection>(combinedStateTag_);
    edm::Ref<reco::TrackInfoCollection>::key_type idti = 0;
    for(traj_iterator=TrajectoryCollection->begin();traj_iterator!=TrajectoryCollection->end();traj_iterator++){//loop on trajectories

      theAlgo_.run(traj_iterator,&rechitscollection,
		   outputFwd,outputBwd,outputUpdated, outputCombined
		   );
      if(outputFwd!=0)outputFwdColl->push_back(*outputFwd);
      if(outputBwd!=0)outputBwdColl->push_back(*outputBwd);
      if(outputUpdated!=0)outputUpdatedColl->push_back(*outputUpdated);
      if(outputCombined!=0)outputCombinedColl->push_back(*outputCombined);

      TrajectoryStateOnSurface outertsos=0;
      TrajectoryStateOnSurface innertsos=0;
      unsigned int outerId=0, innerId=0;
      if (traj_iterator->direction() == alongMomentum) {
	outertsos = traj_iterator->lastMeasurement().updatedState();
	innertsos = traj_iterator->firstMeasurement().updatedState();
	edm::LogInfo("TrackInfoProducer")<<"inner and outer tsos calculated";
	if(traj_iterator->lastMeasurement().recHit()->isValid())outerId = traj_iterator->lastMeasurement().recHit()->geographicalId().rawId();
	else edm::LogInfo("TrackInfoProducer")<<"The hit is not valid";
	if(traj_iterator->firstMeasurement().recHit()->isValid())innerId = traj_iterator->firstMeasurement().recHit()->geographicalId().rawId();
	else edm::LogInfo("TrackInfoProducer")<<"The hit is not valid";
      } else { 
	outertsos = traj_iterator->firstMeasurement().updatedState();
	innertsos = traj_iterator->lastMeasurement().updatedState();
	edm::LogInfo("TrackInfoProducer")<<"inner and outer tsos calculated";
	if(traj_iterator->firstMeasurement().recHit()->isValid())outerId = traj_iterator->firstMeasurement().recHit()->geographicalId().rawId();
	else edm::LogInfo("TrackInfoProducer")<<"The hit is not valid";
	if(traj_iterator->lastMeasurement().recHit()->isValid())innerId = traj_iterator->lastMeasurement().recHit()->geographicalId().rawId();
	else edm::LogInfo("TrackInfoProducer")<<"The hit is not valid";
      }
      edm::LogInfo("TrackInfoProducer")<<"inner id= "<<innerId<<"outer id= "<<outerId;
      if(outerId>0&&innerId>0&& &outertsos!=0  && &innertsos!=0 ){
	GlobalPoint po = outertsos.globalParameters().position();
	GlobalVector vo = outertsos.globalParameters().momentum();
      
	GlobalPoint pi = innertsos.globalParameters().position();
	GlobalVector vi = innertsos.globalParameters().momentum();

     
	reco::TrackCollection::const_iterator tk_iterator;
	edm::Ref<reco::TrackCollection>::key_type idtk = 0;      
	for(tk_iterator=trackCollection->begin();tk_iterator!=trackCollection->end();tk_iterator++){//loop on tracks
	  GlobalPoint tkpo = GlobalPoint(tk_iterator->outerPosition().X(),tk_iterator->outerPosition().Y(),tk_iterator->outerPosition().Z());
	   GlobalVector tkvo = GlobalVector(tk_iterator->outerMomentum().X(),tk_iterator->outerMomentum().Y(),tk_iterator->outerMomentum().Z());
	   GlobalPoint tkpi = GlobalPoint(tk_iterator->innerPosition().X(),tk_iterator->innerPosition().Y(),tk_iterator->innerPosition().Z());
	   GlobalVector tkvi = GlobalVector(tk_iterator->innerMomentum().X(),tk_iterator->innerMomentum().Y(),tk_iterator->innerMomentum().Z());
	   if(((vo-tkvo).mag()<1e-16)&&
	      ((po-tkpo).mag()<1e-16)&&
	      ((vi-tkvi).mag()<1e-16)&&
	      ((pi-tkpi).mag()<1e-16)&&
	      (tk_iterator->innerDetId()==innerId)&&
	      (tk_iterator->outerDetId()==outerId)
	      )
	     {
	       TIassociationFwdColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, idtk),edm::Ref<reco::TrackInfoCollection>(rTrackInfof, idti));
	       TIassociationBwdColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, idtk),edm::Ref<reco::TrackInfoCollection>(rTrackInfob, idti));
	       TIassociationUpdatedColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, idtk),edm::Ref<reco::TrackInfoCollection>(rTrackInfou, idti));
	       TIassociationCombinedColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, idtk),edm::Ref<reco::TrackInfoCollection>(rTrackInfoc, idti));
	     }
	   idtk++;
	}
	idti++;
      }
      else {edm::LogInfo("TrackInfoProducer")<<"association failed"; }
    }
  } 
  catch (cms::Exception &e){ edm::LogInfo("TrackInfoProducer") << "cms::Exception caught!!!" << "\n" << e << "\n";}
  //
  //put everything in the event
  theEvent.put(outputFwdColl,forwardPredictedStateTag_ );
  theEvent.put(outputBwdColl,backwardPredictedStateTag_);
  theEvent.put(outputUpdatedColl,updatedStateTag_ );
  theEvent.put(outputCombinedColl,combinedStateTag_ );

}




