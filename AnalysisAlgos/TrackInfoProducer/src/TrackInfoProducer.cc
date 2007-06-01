#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducer.h"
// system include files
#include <memory>
// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/Common/interface/EDProduct.h" 
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

TrackInfoProducer::TrackInfoProducer(const edm::ParameterSet& iConfig):
    conf_(iConfig),
    theAlgo_(iConfig)
{
  produces<reco::TrackInfoCollection>();
  produces<reco::TrackInfoTrackAssociationCollection>();


}


void TrackInfoProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
   //
  // create empty output collections
  //

  std::auto_ptr<reco::TrackInfoCollection>    outputColl (new reco::TrackInfoCollection);
  std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationColl (new reco::TrackInfoTrackAssociationCollection);
  
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("cosmicTracks");
  
  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  edm::Handle<reco::TrackCollection > trackCollection;
  ///edm::Handle<TrackingRecHitCollection> rechitscollection;
    
    edm::ESHandle<TrackerGeometry> tkgeom;
    setup.get<TrackerDigiGeometryRecord>().get( tkgeom );
    const TrackerGeometry * tracker=&(* tkgeom);
    
  try{  
    
    theEvent.getByLabel(TkTag,TrajectoryCollection);
    theEvent.getByLabel(TkTag,trackCollection);
  } 
  catch (cms::Exception &e){ edm::LogInfo("TrackInfoProducer") << "cms::Exception caught!!!" << "\n" << e << "\n";}
  //
  //run the algorithm  
  //
  reco::TrackInfo output;
  
  std::vector<Trajectory>::const_iterator traj_iterator;
  
  std::vector<unsigned int> trackid;
  for(traj_iterator=TrajectoryCollection->begin();traj_iterator!=TrajectoryCollection->end();++traj_iterator){//loop on trajectories
    //associate the trajectory to the track
    unsigned int idtk = 0;
    reco::TrackRef track;
      bool found=false;
      if(TrajectoryCollection->size()==1){
	trackid.push_back(idtk);
	track=edm::Ref<reco::TrackCollection>(trackCollection, idtk);
	found=true;
      }
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
	     ((pi-tkpi).mag()<1e-16)&&
	     traj_iterator->foundHits()==tk_iterator->found()&&
             traj_iterator->lostHits()==tk_iterator->lost()
	     ){
	    track=edm::Ref<reco::TrackCollection>(trackCollection, idtk);
	    trackid.push_back(idtk);
	    found=true;
	    break;
	  }
	  idtk++;
	}
      }
      
      // build trackinfo
      if(found){
	
	theAlgo_.run(traj_iterator,track,output,tracker);
	
	outputColl->push_back(*(new reco::TrackInfo(output)));
      }
  }

    //put everything in the event
    edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfo;

//     if(forwardPredictedStateTag_!="") rTrackInfof = theEvent.put(outputFwdColl,forwardPredictedStateTag_ );
//     if(backwardPredictedStateTag_!="") rTrackInfob =   theEvent.put(outputBwdColl,backwardPredictedStateTag_);
//     if(updatedStateTag_!="") rTrackInfou =   theEvent.put(outputUpdatedColl,updatedStateTag_ );
//     if(combinedStateTag_!="") rTrackInfoc =   theEvent.put(outputCombinedColl,combinedStateTag_ );
    rTrackInfo=theEvent.put(outputColl);

    for(unsigned int i=0; i <trackid.size();++i){

      TIassociationColl->insert( edm::Ref<reco::TrackCollection>(trackCollection, trackid[i]),edm::Ref<reco::TrackInfoCollection>(rTrackInfo, i));
    }

    theEvent.put(TIassociationColl); 
}




