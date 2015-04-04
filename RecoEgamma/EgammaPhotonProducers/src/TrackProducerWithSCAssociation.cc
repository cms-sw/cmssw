#include "RecoEgamma/EgammaPhotonProducers/interface/TrackProducerWithSCAssociation.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

TrackProducerWithSCAssociation::TrackProducerWithSCAssociation(const edm::ParameterSet& iConfig):
  TrackProducerBase<reco::Track>(iConfig.getParameter<bool>("TrajectoryInEvent")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )), 
          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
          consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );

  if ( iConfig.exists("clusterRemovalInfo") ) {
    edm::InputTag tag = iConfig.getParameter<edm::InputTag>("clusterRemovalInfo");
    if (!(tag == edm::InputTag())) { setClusterRemovalInfo( tag ); }
  }

  
  myname_ = iConfig.getParameter<std::string>("ComponentName");
  conversionTrackCandidateProducer_ = iConfig.getParameter<std::string>("producer");
  trackCSuperClusterAssociationCollection_ = iConfig.getParameter<std::string>("trackCandidateSCAssociationCollection");
  trackSuperClusterAssociationCollection_ = iConfig.getParameter<std::string>("recoTrackSCAssociationCollection");
  myTrajectoryInEvent_ = iConfig.getParameter<bool>("TrajectoryInEvent");

  assoc_token = 
    consumes<reco::TrackCandidateCaloClusterPtrAssociation>(
		    edm::InputTag(conversionTrackCandidateProducer_,
				  trackCSuperClusterAssociationCollection_));
  measurementTrkToken_=
    consumes<MeasurementTrackerEvent>(edm::InputTag("MeasurementTrackerEvent")); //hardcoded because the original was and no time to fix (sigh)
  
 
  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<TrajTrackAssociationCollection>();
  //  produces< reco::TrackSuperClusterAssociationCollection > (trackSuperClusterAssociationCollection_ );
  produces< reco::TrackCaloClusterPtrAssociation > (trackSuperClusterAssociationCollection_ );

}


void TrackProducerWithSCAssociation::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  //edm::LogInfo("TrackProducerWithSCAssociation") << "Analyzing event number: " << theEvent.id() << "\n";

  //LogDebug("TrackProducerWithSCAssociation") << "Analyzing event number: " << theEvent.id() << "\n";
  //  std::cout << " TrackProducerWithSCAssociation Analyzing event number: " << theEvent.id() << "\n";


  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);
  //   Reco Track - Super Cluster Association
  std::auto_ptr<reco::TrackCaloClusterPtrAssociation> scTrkAssoc_p(new reco::TrackCaloClusterPtrAssociation);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

 

  //
  //declare and get TrackColection to be retrieved from the event
  edm::Handle<TrackCandidateCollection> theTCCollection;
  //// Get the association map between candidate out in tracks and the SC where they originated
  validTrackCandidateSCAssociationInput_=true;
  edm::Handle<reco::TrackCandidateCaloClusterPtrAssociation> trkCandidateSCAssocHandle;
  theEvent.getByToken(assoc_token, trkCandidateSCAssocHandle);
  if ( !trkCandidateSCAssocHandle.isValid() ) {
    //    std::cout << "Error! Can't get the product  "<<trackCSuperClusterAssociationCollection_.c_str() << " but keep running. Empty collection will be produced " << "\n";
    edm::LogError("TrackProducerWithSCAssociation") << "Error! Can't get the product  "<<trackCSuperClusterAssociationCollection_.c_str() << " but keep running. Empty collection will be produced " << "\n";
    validTrackCandidateSCAssociationInput_=false;
  }
  reco::TrackCandidateCaloClusterPtrAssociation scTrkCandAssCollection = *(trkCandidateSCAssocHandle.product());
  if ( scTrkCandAssCollection.size() ==0 )  validTrackCandidateSCAssociationInput_=false;


  std::vector<int> tccLocations;
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
  
   
 getFromEvt(theEvent,theTCCollection,bs);  
  
  if (theTCCollection.failedToGet()){
    edm::LogError("TrackProducerWithSCAssociation")  <<"TrackProducerWithSCAssociation could not get the TrackCandidateCollection.";} 
  else{
    //
    //run the algorithm  
    //
    //  LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation run the algorithm" << "\n";
    //    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
    //			     theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
    // we have to copy this method from the algo in order to get the association track-seed
    // this is ugly temporary code that should be replaced!!!!!
    // start of copied code ======================================================
  
    //    std::cout << "TrackProducerWithSCAssociation  Number of TrackCandidates: " << theTCCollection->size() << "\n";
    try{  
      int cont = 0;
      int tcc=0;
   
      for (TrackCandidateCollection::const_iterator i=theTCCollection->begin(); i!=theTCCollection->end();i++)
	{
	  
	  const TrackCandidate * theTC = &(*i);
	  PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
	  const TrackCandidate::range& recHitVec=theTC->recHits();
	  const TrajectorySeed& seed = theTC->seed();
	  
	  //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
	  
	  
	  DetId  detId(state.detId());
	  TrajectoryStateOnSurface theTSOS = trajectoryStateTransform::transientState( state,
									 &(theG.product()->idToDet(detId)->surface()), 
									 theMF.product());
	  
	  //LogDebug("TrackProducerWithSCAssociation")  << "TrackProducerWithSCAssociation  Initial TSOS\n" << theTSOS << "\n";
	  
	  //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
	  //meanwhile computes the number of degrees of freedom
	  TransientTrackingRecHit::RecHitContainer hits;
	  
	  float ndof=0;
	  
	  for (edm::OwnVector<TrackingRecHit>::const_iterator i=recHitVec.first;
	       i!=recHitVec.second; i++){
	    hits.push_back(theBuilder.product()->build(&(*i) ));
	  }

	  
	  //build Track
	  // LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation going to buildTrack"<< "\n";
          FitterCloner fc(theFitter.product(),theBuilder.product());
	  bool ok = theAlgo.buildTrack(fc.fitter.get(),thePropagator.product(),algoResults, hits, theTSOS, seed, ndof, bs, theTC->seedRef());
	  // LogDebug("TrackProducerWithSCAssociation")  << "TrackProducerWithSCAssociation buildTrack result: " << ok << "\n";
	  if(ok) {
	    cont++;
	    tccLocations.push_back(tcc);
	  }
	  tcc++;
	}
      edm::LogInfo("TrackProducerWithSCAssociation") << "Number of Tracks found: " << cont << "\n";
      //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation Number of Tracks found: " << cont << "\n";
      // end of copied code ======================================================
      
    } catch (cms::Exception &e){ edm::LogInfo("TrackProducerWithSCAssociation") << "cms::Exception caught!!!" << "\n" << e << "\n";}
    //
    //put everything in the event
    // we copy putInEvt to get OrphanHandle filled...
    putInEvt(theEvent,thePropagator.product(),theMeasTk.product(), 
	     outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults, theBuilder.product());
    
    // now construct associationmap and put it in the  event
    if (  validTrackCandidateSCAssociationInput_ ) {    
      int itrack=0;
      std::vector<edm::Ptr<reco::CaloCluster> > caloPtrVec;
      for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
	edm::Ref<TrackCandidateCollection> trackCRef(theTCCollection,tccLocations[itrack]);
	const edm::Ptr<reco::CaloCluster>&  aClus = (*trkCandidateSCAssocHandle)[trackCRef];
	caloPtrVec.push_back( aClus );
	itrack++;
      }
      
      
      edm::ValueMap<reco::CaloClusterPtr>::Filler filler(*scTrkAssoc_p);
      filler.insert(rTracks_, caloPtrVec.begin(), caloPtrVec.end());
      filler.fill();
    }    
    
    theEvent.put(scTrkAssoc_p,trackSuperClusterAssociationCollection_ ); 
    
  }

}  
  
std::vector<reco::TransientTrack> TrackProducerWithSCAssociation::getTransient(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackProducerWithSCAssociation") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::vector<reco::TransientTrack> ttks;

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;
 
  try{  
    edm::Handle<TrackCandidateCollection> theTCCollection;
    getFromEvt(theEvent,theTCCollection,bs);
    
    //
    //run the algorithm  
    //
    //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation run the algorithm" << "\n";
   theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			       theFitter.product(), thePropagator.product(), theBuilder.product(), bs, algoResults);

  } catch (cms::Exception &e){ edm::LogInfo("TrackProducerWithSCAssociation") << "cms::Exception caught!!!" << "\n" << e << "\n";}


  for (AlgoProductCollection::iterator prod=algoResults.begin();prod!=algoResults.end(); prod++){
    ttks.push_back( reco::TransientTrack(*(((*prod).second).first),thePropagator.product()->magneticField() ));
  }

  //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation end" << "\n";

  return ttks;
}


#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"


void TrackProducerWithSCAssociation::putInEvt(edm::Event& evt,
					       const Propagator* thePropagator,
					       const MeasurementTracker* theMeasTk,
					       std::auto_ptr<TrackingRecHitCollection>& selHits,
					       std::auto_ptr<reco::TrackCollection>& selTracks,
					       std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
					       std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
					       AlgoProductCollection& algoResults, TransientTrackingRecHitBuilder const * hitBuilder)
{

TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref< std::vector<Trajectory> >::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;

  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(myTrajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
      iTjRef++;
    }
    
    reco::Track * theTrack = (*i).second.first;
    PropagationDirection seedDir = (*i).second.second;
    
    //LogDebug("TrackProducer") << "In KfTrackProducerBase::putInEvt - seedDir=" << seedDir;
    
    reco::Track t = * theTrack;
    selTracks->push_back( t );
    iTkRef++;
    
    // Store indices in local map (starts at 0)
    if(trajectoryInEvent_) tjTkMap[iTjRef-1] = iTkRef-1;
    
    //sets the outermost and innermost TSOSs
    
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    unsigned int innerId, outerId;

    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collision
    if (theTraj->direction() == alongMomentum) {
      outertsos = theTraj->lastMeasurement().updatedState();
      innertsos = theTraj->firstMeasurement().updatedState();
      outerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
    } else {
      outertsos = theTraj->firstMeasurement().updatedState();
      innertsos = theTraj->lastMeasurement().updatedState();
      outerId = theTraj->firstMeasurement().recHit()->geographicalId().rawId();
      innerId = theTraj->lastMeasurement().recHit()->geographicalId().rawId();
    }
    // ---
    //build the TrackExtra
    GlobalPoint v = outertsos.globalParameters().position();
    GlobalVector p = outertsos.globalParameters().momentum();
    math::XYZVector outmom( p.x(), p.y(), p.z() );
    math::XYZPoint  outpos( v.x(), v.y(), v.z() );
    v = innertsos.globalParameters().position();
    p = innertsos.globalParameters().momentum();
    math::XYZVector inmom( p.x(), p.y(), p.z() );
    math::XYZPoint  inpos( v.x(), v.y(), v.z() );

    reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );
    reco::Track & track = selTracks->back();
    track.setExtra( teref );

    //======= I want to set the second hitPattern here =============
    if (theSchool.isValid())
      {
        edm::Handle<MeasurementTrackerEvent> mte;
        evt.getByToken(measurementTrkToken_, mte);
	setSecondHitPattern(theTraj,track,thePropagator,&*mte);
      }
    //==============================================================


    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
                                                 outertsos.curvilinearError(), outerId,
                                                 innertsos.curvilinearError(), innerId,
                                                 seedDir,theTraj->seedRef()));


    reco::TrackExtra & tx = selTrackExtras->back();
   // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collisions
    Traj2TrackHits t2t(hitBuilder,false);
    auto ih = selHits->size();
    assert(ih==hidx);
    t2t(*theTraj,*selHits,false);
    auto const ie = selHits->size();
    unsigned int nHitsAdded = 0;
    for (;ih<ie; ++ih) {
      auto const & hit = (*selHits)[ih];
      track.appendHitPattern(hit);
      ++nHitsAdded;
    }
    tx.setHits( rHits, hidx, nHitsAdded);
    hidx +=nHitsAdded;
    /*
    if (theTraj->direction() == alongMomentum) {
      for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
           j != transHits.end(); j ++ ) {
        if ((**j).hit()!=0){
          TrackingRecHit * hit = (**j).hit()->clone();
          track.appendHitPattern(*hit);
          selHits->push_back( hit );
          tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
        }
      }
    }else{
      for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.end()-1;
           j != transHits.begin()-1; --j ) {
        if ((**j).hit()!=0){
          TrackingRecHit * hit = (**j).hit()->clone();
          track.appendHitPattern(*hit);
          selHits->push_back( hit );
        tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
        }
      }
    }
    */

    // ----

    delete theTrack;
    delete theTraj;
  }

  //LogTrace("TrackingRegressionTest") << "========== TrackProducer Info ===================";
  //LogDebug("TrackProducerWithSCAssociation") << "number of finalTracks: " << selTracks->size() << std::endl;
  //for (reco::TrackCollection::const_iterator it = selTracks->begin(); it != selTracks->end(); it++) {
    //LogDebug("TrackProducerWithSCAssociation")  << "track's n valid and invalid hit, chi2, pt : "
    //                                  << it->found() << " , "
    //                                  << it->lost()  <<" , "
    //                                  << it->normalizedChi2() << " , "
    //	       << it->pt() << std::endl;
  // }
  //LogTrace("TrackingRegressionTest") << "=================================================";


  rTracks_ = evt.put( selTracks );


  evt.put( selTrackExtras );
  evt.put( selHits );
  
  if(myTrajectoryInEvent_) {
    edm::OrphanHandle<std::vector<Trajectory> > rTrajs = evt.put(selTrajectories);
    
    // Now Create traj<->tracks association map
    std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection(rTrajs, rTracks_) );
    for ( std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin();
          i != tjTkMap.end(); i++ ) {
      edm::Ref<std::vector<Trajectory> > trajRef( rTrajs, (*i).first );
      edm::Ref<reco::TrackCollection>    tkRef( rTracks_, (*i).second );
      trajTrackMap->insert( edm::Ref<std::vector<Trajectory> >( rTrajs, (*i).first ),
                            edm::Ref<reco::TrackCollection>( rTracks_, (*i).second ) );
    }
    evt.put( trajTrackMap );
  }
}
