// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      LowPtGsfElectronSeedProducer
//
/**\class LowPtGsfElectronSeedProducer RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronSeedProducer.cc
 Description: EDProducer of ElectronSeed objects
 Implementation:
     <Notes on implementation>
*/
// Original Author:  Robert Bainbridge

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include "LowPtGsfElectronSeedHeavyObjectCache.h"

class LowPtGsfElectronSeedProducer final
    : public edm::stream::EDProducer<edm::GlobalCache<lowptgsfeleseed::HeavyObjectCache> > {
public:
  using TrackIndxMap = std::unordered_map<reco::TrackRef::key_type, size_t>;
  explicit LowPtGsfElectronSeedProducer(const edm::ParameterSet&, const lowptgsfeleseed::HeavyObjectCache*);

  static std::unique_ptr<lowptgsfeleseed::HeavyObjectCache> initializeGlobalCache(const edm::ParameterSet& conf) {
    return std::make_unique<lowptgsfeleseed::HeavyObjectCache>(lowptgsfeleseed::HeavyObjectCache(conf));
  }

  static void globalEndJob(lowptgsfeleseed::HeavyObjectCache const*) {}

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:  // member functions
  template <typename T>
  void loop(const edm::Handle<std::vector<T> >& handle,
            edm::Handle<reco::PFClusterCollection>& hcalClusters,
            reco::ElectronSeedCollection& seeds,
            reco::PreIdCollection& ecalPreIds,
            reco::PreIdCollection& hcalPreIds,
            TrackIndxMap& trksToPreIdIndx,
            edm::Event&,
            const edm::EventSetup&);

  // Overloaded methods to retrieve reco::TrackRef

  reco::TrackRef getBaseRef(edm::Handle<std::vector<reco::Track> > handle, int idx) const;
  reco::TrackRef getBaseRef(edm::Handle<std::vector<reco::PFRecTrack> > handle, int idx) const;

  // Overloaded methods to populate PreIds (using PF or KF tracks)

  void propagateTrackToCalo(const reco::PFRecTrackRef& pfTrackRef,
                            const edm::Handle<reco::PFClusterCollection>& ecalClusters,
                            const edm::Handle<reco::PFClusterCollection>& hcalClusters,
                            std::vector<int>& matchedEcalClusters,
                            std::vector<int>& matchedHcalClusters,
                            reco::PreId& ecalPreId,
                            reco::PreId& hcalPreId);

  void propagateTrackToCalo(const reco::PFRecTrackRef& pfTrackRef,
                            const edm::Handle<reco::PFClusterCollection>& clusters,
                            std::vector<int>& matchedClusters,
                            reco::PreId& preId,
                            bool ecal);

  void propagateTrackToCalo(const reco::TrackRef& pfTrack,
                            const edm::Handle<reco::PFClusterCollection>& ecalClusters,
                            const edm::Handle<reco::PFClusterCollection>& hcalClusters,
                            std::vector<int>& matchedEcalClusters,
                            std::vector<int>& matchedHcalClusters,
                            reco::PreId& ecalPreId,
                            reco::PreId& hcalPreId);
  template <typename CollType>
  void fillPreIdRefValueMap(edm::Handle<CollType> tracksHandle,
                            const TrackIndxMap& trksToPreIdIndx,
                            const edm::OrphanHandle<reco::PreIdCollection>& preIdHandle,
                            edm::ValueMap<reco::PreIdRef>::Filler& filler);

  // Overloaded methods to evaluate BDTs (using PF or KF tracks)

  bool decision(const reco::PFRecTrackRef& pfTrackRef,
                reco::PreId& ecal,
                reco::PreId& hcal,
                double rho,
                const reco::BeamSpot& spot,
                noZS::EcalClusterLazyTools& ecalTools);

  bool decision(const reco::TrackRef& kfTrackRef,
                reco::PreId& ecal,
                reco::PreId& hcal,
                double rho,
                const reco::BeamSpot& spot,
                noZS::EcalClusterLazyTools& ecalTools);

  // Perform lightweight GSF tracking
  bool lightGsfTracking(reco::PreId&, const reco::TrackRef&, const reco::ElectronSeed&);

private:  // member data
  edm::ESHandle<MagneticField> field_;
  std::unique_ptr<TrajectoryFitter> fitterPtr_;
  std::unique_ptr<TrajectorySmoother> smootherPtr_;
  edm::EDGetTokenT<reco::TrackCollection> kfTracks_;
  edm::EDGetTokenT<reco::PFRecTrackCollection> pfTracks_;
  const edm::EDGetTokenT<reco::PFClusterCollection> ecalClusters_;
  edm::EDGetTokenT<reco::PFClusterCollection> hcalClusters_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHits_;
  const edm::EDGetTokenT<double> rho_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpot_;

  const edm::ESGetToken<TrajectoryFitter, TrajectoryFitter::Record> trajectoryFitterToken_;
  const edm::ESGetToken<TrajectorySmoother, TrajectoryFitter::Record> trajectorySmootherToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> builderToken_;
  const noZS::EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  const bool passThrough_;
  const bool usePfTracks_;
  const double minPtThreshold_;
  const double maxPtThreshold_;

  // pow( sinh(1.65), 2. )
  static constexpr double boundary_ = 2.50746495928 * 2.50746495928;
  // pow( ele_mass, 2. )
  static constexpr double mass_ = 0.000511 * 0.000511;
};

//////////////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedProducer::LowPtGsfElectronSeedProducer(const edm::ParameterSet& conf,
                                                           const lowptgsfeleseed::HeavyObjectCache*)
    : field_(),
      fitterPtr_(),
      smootherPtr_(),
      kfTracks_(),
      pfTracks_(),
      ecalClusters_{consumes(conf.getParameter<edm::InputTag>("ecalClusters"))},
      hcalClusters_(),
      ebRecHits_{consumes(conf.getParameter<edm::InputTag>("EBRecHits"))},
      eeRecHits_{consumes(conf.getParameter<edm::InputTag>("EERecHits"))},
      rho_(consumes(conf.getParameter<edm::InputTag>("rho"))),
      beamSpot_(consumes(conf.getParameter<edm::InputTag>("BeamSpot"))),
      trajectoryFitterToken_{esConsumes(conf.getParameter<edm::ESInputTag>("Fitter"))},
      trajectorySmootherToken_{esConsumes(conf.getParameter<edm::ESInputTag>("Smoother"))},
      builderToken_{esConsumes(conf.getParameter<edm::ESInputTag>("TTRHBuilder"))},
      ecalClusterToolsESGetTokens_{consumesCollector()},
      passThrough_(conf.getParameter<bool>("PassThrough")),
      usePfTracks_(conf.getParameter<bool>("UsePfTracks")),
      minPtThreshold_(conf.getParameter<double>("MinPtThreshold")),
      maxPtThreshold_(conf.getParameter<double>("MaxPtThreshold")) {
  if (usePfTracks_) {
    pfTracks_ = consumes(conf.getParameter<edm::InputTag>("pfTracks"));
    hcalClusters_ = consumes(conf.getParameter<edm::InputTag>("hcalClusters"));
  }
  kfTracks_ = consumes(conf.getParameter<edm::InputTag>("tracks"));

  produces<reco::ElectronSeedCollection>();
  produces<reco::PreIdCollection>();
  produces<reco::PreIdCollection>("HCAL");
  produces<edm::ValueMap<reco::PreIdRef> >();  // indexed by edm::Ref<ElectronSeed>.index()
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& setup) {
  setup.get<IdealMagneticFieldRecord>().get(field_);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // Products
  auto seeds = std::make_unique<reco::ElectronSeedCollection>();
  auto ecalPreIds = std::make_unique<reco::PreIdCollection>();
  auto hcalPreIds = std::make_unique<reco::PreIdCollection>();

  const edm::RefProd<reco::PreIdCollection> preIdsRefProd = event.getRefBeforePut<reco::PreIdCollection>();

  // HCAL clusters (only used with PF tracks)
  edm::Handle<reco::PFClusterCollection> hcalClusters;

  //we always need kftracks as we link the preids to them
  edm::Handle<reco::TrackCollection> kfTracks;
  event.getByToken(kfTracks_, kfTracks);

  TrackIndxMap trksToPreIdIndx;
  if (usePfTracks_) {
    edm::Handle<reco::PFRecTrackCollection> pfTracks;
    event.getByToken(pfTracks_, pfTracks);
    event.getByToken(hcalClusters_, hcalClusters);

    //check consistency between kfTracks and pfTracks collection
    for (auto& trk : *pfTracks) {
      if (trk.trackRef().isNonnull()) {
        if (trk.trackRef().id() != kfTracks.id()) {
          throw cms::Exception("ConfigError")
              << "kfTracks is not the collection that pfTracks was built from, please fix this";
        }
        break;  //we only need one valid track ref to check this so end the loop
      }
    }

    loop(pfTracks,  // PF tracks
         hcalClusters,
         *seeds,
         *ecalPreIds,
         *hcalPreIds,
         trksToPreIdIndx,
         event,
         setup);

  } else {
    loop(kfTracks,  // KF tracks
         hcalClusters,
         *seeds,
         *ecalPreIds,
         *hcalPreIds,
         trksToPreIdIndx,
         event,
         setup);
  }

  auto ecalPreIdsHandle = event.put(std::move(ecalPreIds));
  event.put(std::move(hcalPreIds), "HCAL");
  event.put(std::move(seeds));

  auto preIdVMOut = std::make_unique<edm::ValueMap<reco::PreIdRef> >();
  edm::ValueMap<reco::PreIdRef>::Filler mapFiller(*preIdVMOut);
  fillPreIdRefValueMap(kfTracks, trksToPreIdIndx, ecalPreIdsHandle, mapFiller);
  mapFiller.fill();
  event.put(std::move(preIdVMOut));
}

//////////////////////////////////////////////////////////////////////////////////////////
// Return reco::Track from edm::Ref<T>

reco::TrackRef LowPtGsfElectronSeedProducer::getBaseRef(edm::Handle<std::vector<reco::Track> > handle, int idx) const {
  return reco::TrackRef(handle, idx);
}

reco::TrackRef LowPtGsfElectronSeedProducer::getBaseRef(edm::Handle<std::vector<reco::PFRecTrack> > handle,
                                                        int idx) const {
  return reco::PFRecTrackRef(handle, idx)->trackRef();
}

//////////////////////////////////////////////////////////////////////////////////////////
// Template function, instantiated for both reco::Tracks and reco::PFRecTracks
template <typename T>
void LowPtGsfElectronSeedProducer::loop(const edm::Handle<std::vector<T> >& handle,  // PF or KF tracks
                                        edm::Handle<reco::PFClusterCollection>& hcalClusters,
                                        reco::ElectronSeedCollection& seeds,
                                        reco::PreIdCollection& ecalPreIds,
                                        reco::PreIdCollection& hcalPreIds,
                                        TrackIndxMap& trksToPreIdIndx,
                                        edm::Event& event,
                                        const edm::EventSetup& setup) {
  // Pileup
  auto const& rho = event.get(rho_);

  // Beam spot
  auto const& spot = event.get(beamSpot_);

  // Track fitter
  fitterPtr_ = setup.getData(trajectoryFitterToken_).clone();

  // Track smoother
  smootherPtr_.reset(setup.getData(trajectorySmootherToken_).clone());

  // RecHit cloner
  TkClonerImpl hitCloner = static_cast<TkTransientTrackingRecHitBuilder const&>(setup.getData(builderToken_)).cloner();
  fitterPtr_->setHitCloner(&hitCloner);
  smootherPtr_->setHitCloner(&hitCloner);

  // ECAL clusters
  auto ecalClusters = event.getHandle(ecalClusters_);

  // Utility to access to shower shape vars
  noZS::EcalClusterLazyTools ecalTools(event, ecalClusterToolsESGetTokens_.get(setup), ebRecHits_, eeRecHits_);

  // Ensure each cluster is only matched once to a track
  std::vector<int> matchedEcalClusters;
  std::vector<int> matchedHcalClusters;

  // Reserve
  seeds.reserve(handle->size());
  ecalPreIds.reserve(handle->size());
  hcalPreIds.reserve(handle->size());

  // Iterate through (PF or KF) tracks
  for (unsigned int itrk = 0; itrk < handle.product()->size(); itrk++) {
    edm::Ref<std::vector<T> > templatedRef(handle, itrk);  // TrackRef or PFRecTrackRef
    reco::TrackRef trackRef = getBaseRef(handle, itrk);

    if (!(trackRef->quality(reco::TrackBase::qualityByName("highPurity")))) {
      continue;
    }
    if (!passThrough_ && (trackRef->pt() < minPtThreshold_)) {
      continue;
    }

    // Create ElectronSeed
    reco::ElectronSeed seed(*(trackRef->seedRef()));
    seed.setCtfTrack(trackRef);

    // Create PreIds
    unsigned int nModels = globalCache()->modelNames().size();
    reco::PreId ecalPreId(nModels);
    reco::PreId hcalPreId(nModels);

    // Add track ref to PreId
    ecalPreId.setTrack(trackRef);
    hcalPreId.setTrack(trackRef);

    // Add Track-Calo matching variables to PreIds
    propagateTrackToCalo(
        templatedRef, ecalClusters, hcalClusters, matchedEcalClusters, matchedHcalClusters, ecalPreId, hcalPreId);

    // Add variables related to GSF tracks to PreId
    lightGsfTracking(ecalPreId, trackRef, seed);

    // Decision based on BDT
    bool result = decision(templatedRef, ecalPreId, hcalPreId, rho, spot, ecalTools);

    // If fails BDT, do not store seed
    if (!result) {
      continue;
    }

    // Store PreId
    ecalPreIds.push_back(ecalPreId);
    hcalPreIds.push_back(hcalPreId);
    trksToPreIdIndx[trackRef.index()] = ecalPreIds.size() - 1;

    // Store ElectronSeed and corresponding edm::Ref<PreId>.index()
    seeds.push_back(seed);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Template instantiation for reco::Tracks
template void LowPtGsfElectronSeedProducer::loop<reco::Track>(const edm::Handle<std::vector<reco::Track> >&,
                                                              edm::Handle<reco::PFClusterCollection>& hcalClusters,
                                                              reco::ElectronSeedCollection& seeds,
                                                              reco::PreIdCollection& ecalPreIds,
                                                              reco::PreIdCollection& hcalPreIds,
                                                              TrackIndxMap& trksToPreIdIndx,
                                                              edm::Event&,
                                                              const edm::EventSetup&);

//////////////////////////////////////////////////////////////////////////////////////////
// Template instantiation for reco::PFRecTracks
template void LowPtGsfElectronSeedProducer::loop<reco::PFRecTrack>(const edm::Handle<std::vector<reco::PFRecTrack> >&,
                                                                   edm::Handle<reco::PFClusterCollection>& hcalClusters,
                                                                   reco::ElectronSeedCollection& seeds,
                                                                   reco::PreIdCollection& ecalPreIds,
                                                                   reco::PreIdCollection& hcalPreIds,
                                                                   TrackIndxMap& trksToPreIdIndx,
                                                                   edm::Event&,
                                                                   const edm::EventSetup&);

//////////////////////////////////////////////////////////////////////////////////////////
// Loops through both ECAL and HCAL clusters
void LowPtGsfElectronSeedProducer::propagateTrackToCalo(const reco::PFRecTrackRef& pfTrackRef,
                                                        const edm::Handle<reco::PFClusterCollection>& ecalClusters,
                                                        const edm::Handle<reco::PFClusterCollection>& hcalClusters,
                                                        std::vector<int>& matchedEcalClusters,
                                                        std::vector<int>& matchedHcalClusters,
                                                        reco::PreId& ecalPreId,
                                                        reco::PreId& hcalPreId) {
  propagateTrackToCalo(pfTrackRef, ecalClusters, matchedEcalClusters, ecalPreId, true);
  propagateTrackToCalo(pfTrackRef, hcalClusters, matchedHcalClusters, hcalPreId, false);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Loops through ECAL or HCAL clusters (called twice)
void LowPtGsfElectronSeedProducer::propagateTrackToCalo(const reco::PFRecTrackRef& pfTrackRef,
                                                        const edm::Handle<reco::PFClusterCollection>& clusters,
                                                        std::vector<int>& matched,
                                                        reco::PreId& preId,
                                                        bool ecal) {
  // Store info for PreId
  struct Info {
    reco::PFClusterRef cluRef = reco::PFClusterRef();
    float dr2min = 1.e6;
    float deta = 1.e6;
    float dphi = 1.e6;
    math::XYZPoint showerPos = math::XYZPoint(0., 0., 0.);
  } info;

  // Find closest "seed cluster" to KF track extrapolated to ECAL (or HCAL)
  reco::PFTrajectoryPoint point;
  if (ecal) {
    point = pfTrackRef->extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::ECALShowerMax);
  } else {
    point = pfTrackRef->extrapolatedPoint(reco::PFTrajectoryPoint::LayerType::HCALEntrance);
  }

  if (point.isValid()) {
    Info info;
    for (unsigned int iclu = 0; iclu < clusters.product()->size(); iclu++) {
      if (std::find(matched.begin(), matched.end(), iclu) == matched.end()) {
        reco::PFClusterRef cluRef(clusters, iclu);

        // Determine dR squared
        float dr2 = reco::deltaR2(cluRef->positionREP(), point.positionREP());

        if (dr2 < info.dr2min) {
          info.dr2min = dr2;
          info.cluRef = cluRef;
          info.deta = cluRef->positionREP().eta() - point.positionREP().eta();
          info.dphi =
              reco::deltaPhi(cluRef->positionREP().phi(), point.positionREP().phi()) * pfTrackRef->trackRef()->charge();
          info.showerPos = point.position();
        }
      }
    }

    // Set PreId content if match found
    if (info.dr2min < 1.e5) {
      float ep = info.cluRef->correctedEnergy() / std::sqrt(pfTrackRef->trackRef()->innerMomentum().mag2());
      preId.setECALMatchingProperties(info.cluRef,
                                      point.position(),  // ECAL or HCAL surface
                                      info.showerPos,    //
                                      info.deta,
                                      info.dphi,
                                      0.f,                                       // chieta
                                      0.f,                                       // chiphi
                                      pfTrackRef->trackRef()->normalizedChi2(),  // chi2
                                      ep);
    }

  }  // clusters
}

//////////////////////////////////////////////////////////////////////////////////////////
// Original implementation in GoodSeedProducer, loops over ECAL clusters only
void LowPtGsfElectronSeedProducer::propagateTrackToCalo(
    const reco::TrackRef& kfTrackRef,
    const edm::Handle<reco::PFClusterCollection>& ecalClusters,
    const edm::Handle<reco::PFClusterCollection>& hcalClusters,  // not used
    std::vector<int>& matchedEcalClusters,
    std::vector<int>& matchedHcalClusters,  // not used
    reco::PreId& ecalPreId,
    reco::PreId& hcalPreId /* not used */) {
  // Store info for PreId
  struct Info {
    reco::PFClusterRef cluRef = reco::PFClusterRef();
    float dr2min = 1.e6;
    float deta = 1.e6;
    float dphi = 1.e6;
    math::XYZPoint showerPos = math::XYZPoint(0., 0., 0.);
  } info;

  // Propagate 'electron' to ECAL surface
  float energy = sqrt(mass_ + kfTrackRef->outerMomentum().Mag2());
  XYZTLorentzVector mom = XYZTLorentzVector(
      kfTrackRef->outerMomentum().x(), kfTrackRef->outerMomentum().y(), kfTrackRef->outerMomentum().z(), energy);
  XYZTLorentzVector pos = XYZTLorentzVector(
      kfTrackRef->outerPosition().x(), kfTrackRef->outerPosition().y(), kfTrackRef->outerPosition().z(), 0.);
  math::XYZVector field(field_->inTesla(GlobalPoint(0, 0, 0)));
  BaseParticlePropagator particle(RawParticle(mom, pos, kfTrackRef->charge()), 0, 0, field.z());
  particle.propagateToEcalEntrance(false);
  if (particle.getSuccess() == 0) {
    return;
  }

  // ECAL entry point for track
  GlobalPoint ecal_pos(
      particle.particle().vertex().x(), particle.particle().vertex().y(), particle.particle().vertex().z());

  // Preshower limit
  bool below_ps = pow(ecal_pos.z(), 2.) > boundary_ * ecal_pos.perp2();

  // Iterate through ECAL clusters
  for (unsigned int iclu = 0; iclu < ecalClusters.product()->size(); iclu++) {
    reco::PFClusterRef cluRef(ecalClusters, iclu);

    // Correct ecal_pos for shower depth
    double shower_depth = reco::PFCluster::getDepthCorrection(cluRef->correctedEnergy(), below_ps, false);
    GlobalPoint showerPos = ecal_pos + GlobalVector(particle.particle().momentum().x(),
                                                    particle.particle().momentum().y(),
                                                    particle.particle().momentum().z())
                                               .unit() *
                                           shower_depth;

    // Determine dR squared
    float dr2 = reco::deltaR2(cluRef->positionREP(), showerPos);

    // Find nearest ECAL cluster
    if (dr2 < info.dr2min) {
      info.dr2min = dr2;
      info.cluRef = cluRef;
      info.deta = std::abs(cluRef->positionREP().eta() - showerPos.eta());
      info.dphi = std::abs(reco::deltaPhi(cluRef->positionREP().phi(), showerPos.phi())) * kfTrackRef->charge();
      info.showerPos = showerPos;
    }
  }

  // Populate PreId object
  math::XYZPoint point(ecal_pos.x(), ecal_pos.y(), ecal_pos.z());

  // Set PreId content
  ecalPreId.setECALMatchingProperties(
      info.cluRef,
      point,
      info.showerPos,
      info.deta,
      info.dphi,
      0.f,                                                                              // chieta
      0.f,                                                                              // chiphi
      kfTrackRef->normalizedChi2(),                                                     // chi2
      info.cluRef->correctedEnergy() / std::sqrt(kfTrackRef->innerMomentum().mag2()));  // E/p
}

//////////////////////////////////////////////////////////////////////////////////////////
// Original implementation for "lightweight" GSF tracking
bool LowPtGsfElectronSeedProducer::lightGsfTracking(reco::PreId& preId,
                                                    const reco::TrackRef& trackRef,
                                                    const reco::ElectronSeed& seed) {
  Trajectory::ConstRecHitContainer hits;
  for (unsigned int ihit = 0; ihit < trackRef->recHitsSize(); ++ihit) {
    hits.push_back(trackRef->recHit(ihit)->cloneSH());
  }

  GlobalVector gv(trackRef->innerMomentum().x(), trackRef->innerMomentum().y(), trackRef->innerMomentum().z());
  GlobalPoint gp(trackRef->innerPosition().x(), trackRef->innerPosition().y(), trackRef->innerPosition().z());

  GlobalTrajectoryParameters gtps(gp, gv, trackRef->charge(), &*field_);

  TrajectoryStateOnSurface tsos(gtps, trackRef->innerStateCovariance(), *hits[0]->surface());

  // Track fitted and smoothed under electron hypothesis
  Trajectory traj1 = fitterPtr_->fitOne(seed, hits, tsos);
  if (!traj1.isValid()) {
    return false;
  }
  Trajectory traj2 = smootherPtr_->trajectory(traj1);
  if (!traj2.isValid()) {
    return false;
  }

  // Set PreId content
  float chi2Ratio = trackRef->chi2() > 0. ? traj2.chiSquared() / trackRef->chi2() : -1.;
  float gsfReducedChi2 = chi2Ratio > -1. ? chi2Ratio * trackRef->normalizedChi2() : -1.;
  float ptOut = traj2.firstMeasurement().updatedState().globalMomentum().perp();
  float ptIn = traj2.lastMeasurement().updatedState().globalMomentum().perp();
  float gsfDpt = (ptIn > 0) ? fabs(ptOut - ptIn) / ptIn : 0.;
  preId.setTrackProperties(gsfReducedChi2, chi2Ratio, gsfDpt);

  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Decision based on OR of outputs from list of models
bool LowPtGsfElectronSeedProducer::decision(const reco::PFRecTrackRef& pfTrackRef,
                                            reco::PreId& ecalPreId,
                                            reco::PreId& hcalPreId,
                                            double rho,
                                            const reco::BeamSpot& spot,
                                            noZS::EcalClusterLazyTools& ecalTools) {
  bool result = false;
  for (auto& name : globalCache()->modelNames()) {
    result |= globalCache()->eval(name, ecalPreId, hcalPreId, rho, spot, ecalTools);
  }
  return passThrough_ || (pfTrackRef->trackRef()->pt() > maxPtThreshold_) || result;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
bool LowPtGsfElectronSeedProducer::decision(const reco::TrackRef& kfTrackRef,
                                            reco::PreId& ecalPreId,
                                            reco::PreId& hcalPreId,
                                            double rho,
                                            const reco::BeamSpot& spot,
                                            noZS::EcalClusterLazyTools& ecalTools) {
  // No implementation currently
  return passThrough_;
}

template <typename CollType>
void LowPtGsfElectronSeedProducer::fillPreIdRefValueMap(edm::Handle<CollType> tracksHandle,
                                                        const TrackIndxMap& trksToPreIdIndx,
                                                        const edm::OrphanHandle<reco::PreIdCollection>& preIdHandle,
                                                        edm::ValueMap<reco::PreIdRef>::Filler& filler) {
  std::vector<reco::PreIdRef> values;

  unsigned ntracks = tracksHandle->size();
  for (unsigned itrack = 0; itrack < ntracks; ++itrack) {
    edm::Ref<CollType> trackRef(tracksHandle, itrack);
    auto preIdRefIt = trksToPreIdIndx.find(trackRef.index());
    if (preIdRefIt == trksToPreIdIndx.end()) {
      values.push_back(reco::PreIdRef());
    } else {
      edm::Ref<reco::PreIdCollection> preIdRef(preIdHandle, preIdRefIt->second);
      values.push_back(preIdRef);
    }
  }
  filler.insert(tracksHandle, values.begin(), values.end());
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("pfTracks", edm::InputTag("lowPtGsfElePfTracks"));
  desc.add<edm::InputTag>("ecalClusters", edm::InputTag("particleFlowClusterECAL"));
  desc.add<edm::InputTag>("hcalClusters", edm::InputTag("particleFlowClusterHCAL"));
  desc.add<edm::InputTag>("EBRecHits", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHits", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAllTmp"));
  desc.add<edm::InputTag>("BeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::ESInputTag>("Fitter", edm::ESInputTag{"", "GsfTrajectoryFitter_forPreId"});
  desc.add<edm::ESInputTag>("Smoother", edm::ESInputTag{"", "GsfTrajectorySmoother_forPreId"});
  desc.add<edm::ESInputTag>("TTRHBuilder", edm::ESInputTag{"", "WithAngleAndTemplate"});
  desc.add<std::vector<std::string> >("ModelNames", {});
  desc.add<std::vector<std::string> >("ModelWeights", {});
  desc.add<std::vector<double> >("ModelThresholds", {});
  desc.add<bool>("PassThrough", false);
  desc.add<bool>("UsePfTracks", true);
  desc.add<double>("MinPtThreshold", 1.0);
  desc.add<double>("MaxPtThreshold", 15.);
  descriptions.add("lowPtGsfElectronSeeds", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronSeedProducer);
