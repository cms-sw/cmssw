#include "TrackProducerWithSeedAssoc.h"
// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/PixelMatchTrackReco/interface/TrackSeedAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateSeedAssociation.h"

TrackProducerWithSeedAssoc::TrackProducerWithSeedAssoc(const edm::ParameterSet& iConfig):
  TrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent")),
  theAlgo(iConfig)
{
  setConf(iConfig);
  setSrc( iConfig.getParameter<std::string>( "src" ));
  setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
//   string a = alias_;
//   a.erase(a.size()-6,a.size());
  //register your products
  produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
  produces<std::vector<Trajectory> >() ;
  produces<reco::TrackSeedAssociationCollection>();

  assocModule_=iConfig.getParameter<std::string>("src") ;
  //  assocProduct_=iConfig.getParameter<std::string>("AssociationProductName") ;
  myTrajectoryInEvent_=iConfig.getParameter<bool>("TrajectoryInEvent");
}

void TrackProducerWithSeedAssoc::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackProducerWithSeedAssoc") << "Analyzing event number: " << theEvent.id() << "\n";
  //
  // create empty output collections
  //
  std::auto_ptr<TrackingRecHitCollection>    outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection>       outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection>  outputTEColl(new reco::TrackExtraCollection);
  std::auto_ptr<std::vector<Trajectory> >    outputTrajectoryColl(new std::vector<Trajectory>);
  std::auto_ptr<reco::TrackSeedAssociationCollection> outputTSAssCollection(new reco::TrackSeedAssociationCollection);

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  getFromES(setup,theG,theMF,theFitter,thePropagator,theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  edm::Handle<TrackCandidateCollection> theTCCollection;
  edm::Handle<reco::TrackCandidateSeedAssociationCollection> assocTCSeedH;
  theEvent.getByLabel(assocModule_,"",assocTCSeedH);
  reco::TrackCandidateSeedAssociationCollection assocTCSeed=*assocTCSeedH;
  std::vector<int> tccLocations;
  AlgoProductCollection algoResults;
  try{  
    getFromEvt(theEvent,theTCCollection);
    
    //
    //run the algorithm  
    //
    LogDebug("TrackProducerWithSeedAssoc") << "run the algorithm" << "\n";
    //    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
    //			     theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
    // we have to copy this method from the algo in order to get the association track-seed
    // this is ugly temporary code that should be replaced!!!!!
    // start of copied code ======================================================
    edm::LogInfo("TrackProducer") << "Number of TrackCandidates: " << theTCCollection->size() << "\n";

    int cont = 0;
    int tcc=0;
    for (TrackCandidateCollection::const_iterator i=theTCCollection->begin(); i!=theTCCollection->end();i++)
      {
      
	const TrackCandidate * theTC = &(*i);
	PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
	const TrackCandidate::range& recHitVec=theTC->recHits();
	const TrajectorySeed& seed = theTC->seed();

	//convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
	TrajectoryStateTransform transformer;
  
	DetId  detId(state.detId());
	TrajectoryStateOnSurface theTSOS = transformer.transientState( state,
								       &(theG.product()->idToDet(detId)->surface()), 
								       theMF.product());

	LogDebug("TrackProducer") << "Initial TSOS\n" << theTSOS << "\n";
      
	//convert the TrackingRecHit vector to a TransientTrackingRecHit vector
	//meanwhile computes the number of degrees of freedom
	TransientTrackingRecHit::RecHitContainer hits;
      
	float ndof=0;
      
	for (edm::OwnVector<TrackingRecHit>::const_iterator i=recHitVec.first;
	     i!=recHitVec.second; i++){
	  hits.push_back(theBuilder.product()->build(&(*i) ));
	  if ((*i).isValid()){
	    ndof = ndof + (i->dimension())*(i->weight());
	  }
	}
      
	ndof = ndof - 5;
      
	//build Track
	LogDebug("TrackProducer") << "going to buildTrack"<< "\n";
	bool ok = theAlgo.buildTrack(theFitter.product(),thePropagator.product(),algoResults, hits, theTSOS, seed, ndof);
	LogDebug("TrackProducer") << "buildTrack result: " << ok << "\n";
	if(ok) {
	  cont++;
	  tccLocations.push_back(tcc);
	}
	tcc++;
      }
    edm::LogInfo("TrackProducerWithSeedAssoc") << "Number of Tracks found: " << cont << "\n";
    // end of copied code ======================================================

  } catch (cms::Exception &e){ edm::LogInfo("TrackProducerWithSeedAssoc") << "cms::Exception caught!!!" << "\n" << e << "\n";}
  //
  //put everything in the event
  // we copy putInEvt to get OrphanHandle filled...
  putInEvt(theEvent, outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl, algoResults);

  // now construct associationmap and put it into event
  int itrack=0;
  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    edm::Ref<reco::TrackCollection> trackRef(rTracks_,itrack);
    edm::Ref<TrackCandidateCollection> trackCRef(theTCCollection,tccLocations[itrack]);
    edm::Ref<TrajectorySeedCollection> seedRef= assocTCSeed[trackCRef];
    outputTSAssCollection->insert(trackRef,seedRef);
    itrack++;
  }
  theEvent.put(outputTSAssCollection);

  LogDebug("TrackProducerWithSeedAssoc") << "end" << "\n";
}


std::vector<reco::TransientTrack> TrackProducerWithSeedAssoc::getTransient(edm::Event& theEvent, const edm::EventSetup& setup)
{
  edm::LogInfo("TrackProducerWithSeedAssoc") << "Analyzing event number: " << theEvent.id() << "\n";
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
  getFromES(setup,theG,theMF,theFitter,thePropagator,theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  try{  
    edm::Handle<TrackCandidateCollection> theTCCollection;
    getFromEvt(theEvent,theTCCollection);
    
    //
    //run the algorithm  
    //
    LogDebug("TrackProducerWithSeedAssoc") << "run the algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(), theMF.product(), *theTCCollection, 
			     theFitter.product(), thePropagator.product(), theBuilder.product(), algoResults);
  } catch (cms::Exception &e){ edm::LogInfo("TrackProducerWithSeedAssoc") << "cms::Exception caught!!!" << "\n" << e << "\n";}


  for (AlgoProductCollection::iterator prod=algoResults.begin();prod!=algoResults.end(); prod++){
    ttks.push_back( reco::TransientTrack(*(((*prod).second).first),thePropagator.product()->magneticField() ));
  }

  LogDebug("TrackProducerWithSeedAssoc") << "end" << "\n";

  return ttks;
}


// this code had to be copied to get the OrphanHandle
// without that the insert of the trackRef into the AssociationMap fails
//very temporary, should be changed in the future!!!

 void TrackProducerWithSeedAssoc::putInEvt(edm::Event& evt,
				 std::auto_ptr<TrackingRecHitCollection>& selHits,
				 std::auto_ptr<reco::TrackCollection>& selTracks,
				 std::auto_ptr<reco::TrackExtraCollection>& selTrackExtras,
				 std::auto_ptr<std::vector<Trajectory> >&   selTrajectories,
				 AlgoProductCollection& algoResults)
{

  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
  //doesnt work!!  reco::TrackRefProd rTracks = evt.getRefBeforePut<reco::TrackCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackExtraCollection>::key_type hidx = 0;
  for(AlgoProductCollection::iterator i=algoResults.begin(); i!=algoResults.end();i++){
    Trajectory * theTraj = (*i).first;
    if(myTrajectoryInEvent_) selTrajectories->push_back(*theTraj);
    const TrajectoryFitter::RecHitContainer& transHits = theTraj->recHits();

    reco::Track * theTrack = ((*i).second).first;
    
    //     if( ) {
    reco::Track t = * theTrack;
    selTracks->push_back( t );
    
    //sets the outermost and innermost TSOSs
    TrajectoryStateOnSurface outertsos;
    TrajectoryStateOnSurface innertsos;
    unsigned int innerId, outerId;
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
    PropagationDirection dir = theTraj->direction();
    selTrackExtras->push_back( reco::TrackExtra (outpos, outmom, true, inpos, inmom, true,
						 outertsos.curvilinearError(), outerId,
						 innertsos.curvilinearError(), innerId, dir));

    reco::TrackExtra & tx = selTrackExtras->back();
    size_t k = 0;
    for( TrajectoryFitter::RecHitContainer::const_iterator j = transHits.begin();
	 j != transHits.end(); j ++ ) {
      TrackingRecHit * hit = (**j).hit()->clone();
      track.setHitPattern( * hit, k ++ );
      selHits->push_back( hit );
      tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
    }
    delete theTrack;
    delete theTraj;
  }

  rTracks_ = evt.put( selTracks ); //this is changed
  evt.put( selTrackExtras );
  evt.put( selHits );
  if(myTrajectoryInEvent_) evt.put(selTrajectories);
}

