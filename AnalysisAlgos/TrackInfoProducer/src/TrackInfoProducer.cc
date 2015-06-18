#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducer.h"
// system include files
#include <memory>
// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
TrackInfoProducer::TrackInfoProducer(const edm::ParameterSet& iConfig):
    theAlgo_(iConfig),
    TrajectoryToken_(consumes<std::vector<Trajectory> >(iConfig.getParameter<edm::InputTag>("cosmicTracks"))),
    trackCollectionToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("cosmicTracks"))),
    assoMapToken_(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("cosmicTracks")))
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

  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  edm::Handle<reco::TrackCollection > trackCollection;
  edm::Handle<TrajTrackAssociationCollection> assoMap;

  edm::ESHandle<TrackerGeometry> tkgeom;
  setup.get<TrackerDigiGeometryRecord>().get( tkgeom );
  const TrackerGeometry * tracker=&(* tkgeom);

  theEvent.getByToken(TrajectoryToken_,TrajectoryCollection);
  theEvent.getByToken(trackCollectionToken_,trackCollection);
  theEvent.getByToken(assoMapToken_,assoMap);

  //
  //run the algorithm
  //
  reco::TrackInfo output;

  std::vector<Trajectory>::const_iterator traj_iterator;
  edm::LogInfo("TrackInfoProducer") << "Loop on trajectories";
  std::map<reco::TrackRef,unsigned int> trackid;
  int i=0;

 for(TrajTrackAssociationCollection::const_iterator it = assoMap->begin();it != assoMap->end(); ++it){
   const edm::Ref<std::vector<Trajectory> > traj = it->key;
   const reco::TrackRef track = it->val;
   trackid.insert(make_pair(track,i));
   i++;
   theAlgo_.run(traj,track,output,tracker);
   outputColl->push_back(*(new reco::TrackInfo(output)));

 }


    //put everything in the event
    edm::OrphanHandle<reco::TrackInfoCollection> rTrackInfo;

//     if(forwardPredictedStateTag_!="") rTrackInfof = theEvent.put(outputFwdColl,forwardPredictedStateTag_ );
//     if(backwardPredictedStateTag_!="") rTrackInfob =   theEvent.put(outputBwdColl,backwardPredictedStateTag_);
//     if(updatedStateTag_!="") rTrackInfou =   theEvent.put(outputUpdatedColl,updatedStateTag_ );
//     if(combinedStateTag_!="") rTrackInfoc =   theEvent.put(outputCombinedColl,combinedStateTag_ );
    rTrackInfo=theEvent.put(outputColl);
    std::auto_ptr<reco::TrackInfoTrackAssociationCollection>    TIassociationColl (new reco::TrackInfoTrackAssociationCollection(assoMap->refProd().val, rTrackInfo));

    for(std::map<reco::TrackRef,unsigned int>::iterator ref_iter=trackid.begin();ref_iter!=trackid.end();ref_iter++){

      TIassociationColl->insert( ref_iter->first,edm::Ref<reco::TrackInfoCollection>(rTrackInfo,ref_iter->second ));
    }

    theEvent.put(TIassociationColl);
}




