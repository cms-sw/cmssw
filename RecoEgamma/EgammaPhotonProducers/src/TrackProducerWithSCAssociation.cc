/** \class  TrackProducerWithSCAssociation
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **   Modified version of TrackProducer by Giuseppe Cerati
 **   to have super cluster - conversion track association
 ** 
 ***/

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

class TrackProducerWithSCAssociation : public TrackProducerBase<reco::Track>, public edm::stream::EDProducer<> {
public:
  explicit TrackProducerWithSCAssociation(const edm::ParameterSet& iConfig);

  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  std::string myname_;
  TrackProducerAlgorithm<reco::Track> theAlgo;
  std::string conversionTrackCandidateProducer_;
  std::string trackCSuperClusterAssociationCollection_;
  std::string trackSuperClusterAssociationCollection_;
  edm::EDGetTokenT<reco::TrackCandidateCaloClusterPtrAssociation> assoc_token;
  edm::OrphanHandle<reco::TrackCollection> rTracks_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrkToken_;
  bool myTrajectoryInEvent_;
  bool validTrackCandidateSCAssociationInput_;

  //Same recipe as Ursula's for electrons. Copy this from TrackProducerBase to get the OrphanHandle
  //ugly temporary solution!! I agree !
  void putInEvt(edm::Event& evt,
                const Propagator* thePropagator,
                const MeasurementTracker* theMeasTk,
                std::unique_ptr<TrackingRecHitCollection> selHits,
                std::unique_ptr<reco::TrackCollection> selTracks,
                std::unique_ptr<reco::TrackExtraCollection> selTrackExtras,
                std::unique_ptr<std::vector<Trajectory>> selTrajectories,
                AlgoProductCollection& algoResults,
                TransientTrackingRecHitBuilder const* hitBuilder,
                const TrackerTopology* ttopo);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackProducerWithSCAssociation);

TrackProducerWithSCAssociation::TrackProducerWithSCAssociation(const edm::ParameterSet& iConfig)
    : TrackProducerBase<reco::Track>(iConfig.getParameter<bool>("TrajectoryInEvent")), theAlgo(iConfig) {
  setConf(iConfig);
  setSrc(consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>("src")),
         consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot")),
         consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent")));
  setAlias(iConfig.getParameter<std::string>("@module_label"));

  if (iConfig.exists("clusterRemovalInfo")) {
    edm::InputTag tag = iConfig.getParameter<edm::InputTag>("clusterRemovalInfo");
    if (!(tag == edm::InputTag())) {
      setClusterRemovalInfo(tag);
    }
  }

  myname_ = iConfig.getParameter<std::string>("ComponentName");
  conversionTrackCandidateProducer_ = iConfig.getParameter<std::string>("producer");
  trackCSuperClusterAssociationCollection_ = iConfig.getParameter<std::string>("trackCandidateSCAssociationCollection");
  trackSuperClusterAssociationCollection_ = iConfig.getParameter<std::string>("recoTrackSCAssociationCollection");
  myTrajectoryInEvent_ = iConfig.getParameter<bool>("TrajectoryInEvent");

  assoc_token = consumes<reco::TrackCandidateCaloClusterPtrAssociation>(
      edm::InputTag(conversionTrackCandidateProducer_, trackCSuperClusterAssociationCollection_));
  measurementTrkToken_ = consumes<MeasurementTrackerEvent>(
      edm::InputTag("MeasurementTrackerEvent"));  //hardcoded because the original was and no time to fix (sigh)

  //register your products
  produces<reco::TrackCollection>().setBranchAlias(alias_ + "Tracks");
  produces<reco::TrackExtraCollection>().setBranchAlias(alias_ + "TrackExtras");
  produces<TrackingRecHitCollection>().setBranchAlias(alias_ + "RecHits");
  produces<std::vector<Trajectory>>();
  produces<TrajTrackAssociationCollection>();
  //  produces< reco::TrackSuperClusterAssociationCollection > (trackSuperClusterAssociationCollection_ );
  produces<reco::TrackCaloClusterPtrAssociation>(trackSuperClusterAssociationCollection_);
}

void TrackProducerWithSCAssociation::produce(edm::Event& theEvent, const edm::EventSetup& setup) {
  //edm::LogInfo("TrackProducerWithSCAssociation") << "Analyzing event number: " << theEvent.id() << "\n";

  //LogDebug("TrackProducerWithSCAssociation") << "Analyzing event number: " << theEvent.id() << "\n";
  //  std::cout << " TrackProducerWithSCAssociation Analyzing event number: " << theEvent.id() << "\n";

  //
  // create empty output collections
  //
  auto outputRHColl = std::make_unique<TrackingRecHitCollection>();
  auto outputTColl = std::make_unique<reco::TrackCollection>();
  auto outputTEColl = std::make_unique<reco::TrackExtraCollection>();
  auto outputTrajectoryColl = std::make_unique<std::vector<Trajectory>>();
  //   Reco Track - Super Cluster Association
  auto scTrkAssoc_p = std::make_unique<reco::TrackCaloClusterPtrAssociation>();

  //
  //declare and get stuff to be retrieved from ES
  //
  edm::ESHandle<TrackerGeometry> theG;
  edm::ESHandle<MagneticField> theMF;
  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology* ttopo = httopo.product();

  //
  //declare and get TrackColection to be retrieved from the event
  edm::Handle<TrackCandidateCollection> theTCCollection;
  //// Get the association map between candidate out in tracks and the SC where they originated
  validTrackCandidateSCAssociationInput_ = true;
  edm::Handle<reco::TrackCandidateCaloClusterPtrAssociation> trkCandidateSCAssocHandle;
  theEvent.getByToken(assoc_token, trkCandidateSCAssocHandle);
  if (!trkCandidateSCAssocHandle.isValid()) {
    //    std::cout << "Error! Can't get the product  "<<trackCSuperClusterAssociationCollection_.c_str() << " but keep running. Empty collection will be produced " << "\n";
    edm::LogError("TrackProducerWithSCAssociation")
        << "Error! Can't get the product  " << trackCSuperClusterAssociationCollection_.c_str()
        << " but keep running. Empty collection will be produced "
        << "\n";
    validTrackCandidateSCAssociationInput_ = false;
  }
  reco::TrackCandidateCaloClusterPtrAssociation scTrkCandAssCollection = *(trkCandidateSCAssocHandle.product());
  if (scTrkCandAssCollection.empty())
    validTrackCandidateSCAssociationInput_ = false;

  std::vector<int> tccLocations;
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;

  getFromEvt(theEvent, theTCCollection, bs);

  if (theTCCollection.failedToGet()) {
    edm::LogError("TrackProducerWithSCAssociation")
        << "TrackProducerWithSCAssociation could not get the TrackCandidateCollection.";
  } else {
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
    try {
      int cont = 0;
      int tcc = 0;

      for (TrackCandidateCollection::const_iterator i = theTCCollection->begin(); i != theTCCollection->end(); i++) {
        const TrackCandidate* theTC = &(*i);
        PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
        const TrajectorySeed& seed = theTC->seed();

        //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface

        DetId detId(state.detId());
        TrajectoryStateOnSurface theTSOS = trajectoryStateTransform::transientState(
            state, &(theG.product()->idToDet(detId)->surface()), theMF.product());

        //LogDebug("TrackProducerWithSCAssociation")  << "TrackProducerWithSCAssociation  Initial TSOS\n" << theTSOS << "\n";

        //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
        //meanwhile computes the number of degrees of freedom
        TransientTrackingRecHit::RecHitContainer hits;

        float ndof = 0;

        for (auto const& recHit : theTC->recHits()) {
          hits.push_back(theBuilder.product()->build(&recHit));
        }

        //build Track
        // LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation going to buildTrack"<< "\n";
        FitterCloner fc(theFitter.product(), theBuilder.product());
        bool ok = theAlgo.buildTrack(
            fc.fitter.get(), thePropagator.product(), algoResults, hits, theTSOS, seed, ndof, bs, theTC->seedRef());
        // LogDebug("TrackProducerWithSCAssociation")  << "TrackProducerWithSCAssociation buildTrack result: " << ok << "\n";
        if (ok) {
          cont++;
          tccLocations.push_back(tcc);
        }
        tcc++;
      }
      edm::LogInfo("TrackProducerWithSCAssociation") << "Number of Tracks found: " << cont << "\n";
      //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation Number of Tracks found: " << cont << "\n";
      // end of copied code ======================================================

    } catch (cms::Exception& e) {
      edm::LogInfo("TrackProducerWithSCAssociation") << "cms::Exception caught!!!"
                                                     << "\n"
                                                     << e << "\n";
    }
    //
    //put everything in the event
    // we copy putInEvt to get OrphanHandle filled...
    putInEvt(theEvent,
             thePropagator.product(),
             theMeasTk.product(),
             std::move(outputRHColl),
             std::move(outputTColl),
             std::move(outputTEColl),
             std::move(outputTrajectoryColl),
             algoResults,
             theBuilder.product(),
             ttopo);

    // now construct associationmap and put it in the  event
    if (validTrackCandidateSCAssociationInput_) {
      int itrack = 0;
      std::vector<edm::Ptr<reco::CaloCluster>> caloPtrVec;
      for (AlgoProductCollection::iterator i = algoResults.begin(); i != algoResults.end(); i++) {
        edm::Ref<TrackCandidateCollection> trackCRef(theTCCollection, tccLocations[itrack]);
        const edm::Ptr<reco::CaloCluster>& aClus = (*trkCandidateSCAssocHandle)[trackCRef];
        caloPtrVec.push_back(aClus);
        itrack++;
      }

      edm::ValueMap<reco::CaloClusterPtr>::Filler filler(*scTrkAssoc_p);
      filler.insert(rTracks_, caloPtrVec.begin(), caloPtrVec.end());
      filler.fill();
    }

    theEvent.put(std::move(scTrkAssoc_p), trackSuperClusterAssociationCollection_);
  }
}

std::vector<reco::TransientTrack> TrackProducerWithSCAssociation::getTransient(edm::Event& theEvent,
                                                                               const edm::EventSetup& setup) {
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
  getFromES(setup, theG, theMF, theFitter, thePropagator, theMeasTk, theBuilder);

  //
  //declare and get TrackColection to be retrieved from the event
  //
  AlgoProductCollection algoResults;
  reco::BeamSpot bs;

  try {
    edm::Handle<TrackCandidateCollection> theTCCollection;
    getFromEvt(theEvent, theTCCollection, bs);

    //
    //run the algorithm
    //
    //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation run the algorithm" << "\n";
    theAlgo.runWithCandidate(theG.product(),
                             theMF.product(),
                             *theTCCollection,
                             theFitter.product(),
                             thePropagator.product(),
                             theBuilder.product(),
                             bs,
                             algoResults);

  } catch (cms::Exception& e) {
    edm::LogInfo("TrackProducerWithSCAssociation") << "cms::Exception caught!!!"
                                                   << "\n"
                                                   << e << "\n";
  }

  for (auto& prod : algoResults) {
    ttks.emplace_back(*prod.track, thePropagator.product()->magneticField());
  }

  //LogDebug("TrackProducerWithSCAssociation") << "TrackProducerWithSCAssociation end" << "\n";

  return ttks;
}

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"

void TrackProducerWithSCAssociation::putInEvt(edm::Event& evt,
                                              const Propagator* thePropagator,
                                              const MeasurementTracker* theMeasTk,
                                              std::unique_ptr<TrackingRecHitCollection> selHits,
                                              std::unique_ptr<reco::TrackCollection> selTracks,
                                              std::unique_ptr<reco::TrackExtraCollection> selTrackExtras,
                                              std::unique_ptr<std::vector<Trajectory>> selTrajectories,
                                              AlgoProductCollection& algoResults,
                                              TransientTrackingRecHitBuilder const* hitBuilder,
                                              const TrackerTopology* ttopo) {
  TrackingRecHitRefProd rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();

  edm::Ref<reco::TrackExtraCollection>::key_type idx = 0;
  edm::Ref<reco::TrackCollection>::key_type iTkRef = 0;
  edm::Ref<std::vector<Trajectory>>::key_type iTjRef = 0;
  std::map<unsigned int, unsigned int> tjTkMap;

  for (auto& i : algoResults) {
    Trajectory* theTraj = i.trajectory;
    if (myTrajectoryInEvent_) {
      selTrajectories->push_back(*theTraj);
      iTjRef++;
    }

    reco::Track* theTrack = i.track;
    PropagationDirection seedDir = i.pDir;

    //LogDebug("TrackProducer") << "In KfTrackProducerBase::putInEvt - seedDir=" << seedDir;

    reco::Track t = *theTrack;
    selTracks->push_back(t);
    iTkRef++;

    // Store indices in local map (starts at 0)
    if (trajectoryInEvent_)
      tjTkMap[iTjRef - 1] = iTkRef - 1;

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
    math::XYZVector outmom(p.x(), p.y(), p.z());
    math::XYZPoint outpos(v.x(), v.y(), v.z());
    v = innertsos.globalParameters().position();
    p = innertsos.globalParameters().momentum();
    math::XYZVector inmom(p.x(), p.y(), p.z());
    math::XYZPoint inpos(v.x(), v.y(), v.z());

    reco::TrackExtraRef teref = reco::TrackExtraRef(rTrackExtras, idx++);
    reco::Track& track = selTracks->back();
    track.setExtra(teref);

    //======= I want to set the second hitPattern here =============
    if (theSchool.isValid()) {
      edm::Handle<MeasurementTrackerEvent> mte;
      evt.getByToken(measurementTrkToken_, mte);
      setSecondHitPattern(theTraj, track, thePropagator, &*mte, ttopo);
    }
    //==============================================================

    selTrackExtras->push_back(reco::TrackExtra(outpos,
                                               outmom,
                                               true,
                                               inpos,
                                               inmom,
                                               true,
                                               outertsos.curvilinearError(),
                                               outerId,
                                               innertsos.curvilinearError(),
                                               innerId,
                                               seedDir,
                                               theTraj->seedRef()));

    reco::TrackExtra& tx = selTrackExtras->back();
    // ---  NOTA BENE: the convention is to sort hits and measurements "along the momentum".
    // This is consistent with innermost and outermost labels only for tracks from LHC collisions
    reco::TrackExtra::TrajParams trajParams;
    reco::TrackExtra::Chi2sFive chi2s;
    Traj2TrackHits t2t;
    auto ih = selHits->size();
    t2t(*theTraj, *selHits, trajParams, chi2s);
    auto ie = selHits->size();
    tx.setHits(rHits, ih, ie - ih);
    tx.setTrajParams(std::move(trajParams), std::move(chi2s));
    for (; ih < ie; ++ih) {
      auto const& hit = (*selHits)[ih];
      track.appendHitPattern(hit, *ttopo);
    }
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

  rTracks_ = evt.put(std::move(selTracks));

  evt.put(std::move(selTrackExtras));
  evt.put(std::move(selHits));

  if (myTrajectoryInEvent_) {
    edm::OrphanHandle<std::vector<Trajectory>> rTrajs = evt.put(std::move(selTrajectories));

    // Now Create traj<->tracks association map
    auto trajTrackMap = std::make_unique<TrajTrackAssociationCollection>(rTrajs, rTracks_);
    for (std::map<unsigned int, unsigned int>::iterator i = tjTkMap.begin(); i != tjTkMap.end(); i++) {
      edm::Ref<std::vector<Trajectory>> trajRef(rTrajs, (*i).first);
      edm::Ref<reco::TrackCollection> tkRef(rTracks_, (*i).second);
      trajTrackMap->insert(edm::Ref<std::vector<Trajectory>>(rTrajs, (*i).first),
                           edm::Ref<reco::TrackCollection>(rTracks_, (*i).second));
    }
    evt.put(std::move(trajTrackMap));
  }
}
