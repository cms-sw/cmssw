#include "TrackMerger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#define TRACK_SORT 1  // just use all hits from inner track, then append from outer outside it
#define DET_SORT 0    // sort hits using global position of detector
#define HIT_SORT 0    // sort hits using global position of hit

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #define VI_DEBUG

#ifdef VI_DEBUG
#define DPRINT(x) std::cout << x << ": "
#define PRINT std::cout
#else
#define DPRINT(x) LogTrace(x)
#define PRINT LogTrace("")
#endif

TrackMerger::TrackMerger(const edm::ParameterSet &iConfig, edm::ConsumesCollector cc)
    : useInnermostState_(iConfig.getParameter<bool>("useInnermostState")),
      debug_(false),
      theBuilderName(iConfig.getParameter<std::string>("ttrhBuilderName")),
      geometryToken_(cc.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      magFieldToken_(cc.esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      builderToken_(
          cc.esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(edm::ESInputTag("", theBuilderName))),
      trackerTopoToken_(cc.esConsumes<TrackerTopology, TrackerTopologyRcd>()) {}

TrackMerger::~TrackMerger() {}

void TrackMerger::init(const edm::EventSetup &iSetup) {
  theGeometry = iSetup.getHandle(geometryToken_);
  theMagField = iSetup.getHandle(magFieldToken_);
  theBuilder = iSetup.getHandle(builderToken_);
  theTrkTopo = iSetup.getHandle(trackerTopoToken_);
}

TrackCandidate TrackMerger::merge(const reco::Track &inner,
                                  const reco::Track &outer,
                                  DuplicateTrackType duplicateType) const {
  DPRINT("TrackMerger") << std::abs(inner.eta()) << " merging " << inner.algo() << '/' << outer.algo() << ' '
                        << inner.eta() << '/' << outer.eta() << std::endl;

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(inner.recHitsSize() + outer.recHitsSize());
  DPRINT("TrackMerger") << "Inner track hits: " << std::endl;
  for (auto it = inner.recHitsBegin(), ed = inner.recHitsEnd(); it != ed; ++it) {
    hits.push_back(&**it);
    if (debug_) {
      DetId id(hits.back()->geographicalId());
      PRINT << "   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid "
            << hits.back()->isValid() << "   detid: " << id() << std::endl;
    }
  }
  DPRINT("TrackMerger") << "Outer track hits: " << std::endl;

  if (duplicateType == DuplicateTrackType::Disjoint) {
#if TRACK_SORT
    DetId lastId(hits.back()->geographicalId());
    int lastSubdet = lastId.subdetId();
    unsigned int lastLayer = theTrkTopo->layer(lastId);
    for (auto it = outer.recHitsBegin(), ed = outer.recHitsEnd(); it != ed; ++it) {
      const TrackingRecHit *hit = &**it;
      DetId id(hit->geographicalId());
      int thisSubdet = id.subdetId();
      if (thisSubdet > lastSubdet || (thisSubdet == lastSubdet && theTrkTopo->layer(id) > lastLayer)) {
        hits.push_back(hit);
        PRINT << "  adding   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid "
              << hit->isValid() << "   detid: " << id() << std::endl;
      } else {
        PRINT << "  skipping subdet " << thisSubdet << "  layer " << theTrkTopo->layer(id) << " valid "
              << hit->isValid() << "   detid: " << id() << std::endl;
      }
    }
#else
    addSecondTrackHits(hits, outer);
#endif
  } else if (duplicateType == DuplicateTrackType::Overlapping) {
    addSecondTrackHits(hits, outer);
  }

  math::XYZVector p = (inner.innerMomentum() + outer.outerMomentum());
  GlobalVector v(p.x(), p.y(), p.z());
  if (!useInnermostState_)
    v *= -1;

  TrackCandidate::RecHitContainer ownHits;
  unsigned int nhits = hits.size();
  ownHits.reserve(nhits);

  if (duplicateType == DuplicateTrackType::Disjoint) {
#if TRACK_SORT
    if (!useInnermostState_)
      std::reverse(hits.begin(), hits.end());
    for (auto hit : hits)
      ownHits.push_back(*hit);
#elif DET_SORT
    // OLD METHOD, sometimes fails
    std::sort(hits.begin(), hits.end(), MomentumSort(v, &*theGeometry));
    for (auto hit : hits)
      ownHits.push_back(*hit);
#else
    sortByHitPosition(v, hits, ownHits);
#endif
  } else if (duplicateType == DuplicateTrackType::Overlapping) {
    sortByHitPosition(v, hits, ownHits);
  }

  PTrajectoryStateOnDet state;
  PropagationDirection pdir;
  if (useInnermostState_) {
    pdir = alongMomentum;
    if ((inner.outerPosition() - inner.innerPosition()).Dot(inner.momentum()) >= 0) {
      // use inner state
      TrajectoryStateOnSurface originalTsosIn(
          trajectoryStateTransform::innerStateOnSurface(inner, *theGeometry, &*theMagField));
      state = trajectoryStateTransform::persistentState(originalTsosIn, DetId(inner.innerDetId()));
    } else {
      // use outer state
      TrajectoryStateOnSurface originalTsosOut(
          trajectoryStateTransform::outerStateOnSurface(inner, *theGeometry, &*theMagField));
      state = trajectoryStateTransform::persistentState(originalTsosOut, DetId(inner.outerDetId()));
    }
  } else {
    pdir = oppositeToMomentum;
    if ((outer.outerPosition() - inner.innerPosition()).Dot(inner.momentum()) >= 0) {
      // use outer state
      TrajectoryStateOnSurface originalTsosOut(
          trajectoryStateTransform::outerStateOnSurface(outer, *theGeometry, &*theMagField));
      state = trajectoryStateTransform::persistentState(originalTsosOut, DetId(outer.outerDetId()));
    } else {
      // use inner state
      TrajectoryStateOnSurface originalTsosIn(
          trajectoryStateTransform::innerStateOnSurface(outer, *theGeometry, &*theMagField));
      state = trajectoryStateTransform::persistentState(originalTsosIn, DetId(outer.innerDetId()));
    }
  }
  TrajectorySeed seed(state, TrackCandidate::RecHitContainer(), pdir);
  TrackCandidate ret(ownHits, seed, state, (useInnermostState_ ? inner : outer).seedRef());
  ret.setStopReason((uint8_t)(useInnermostState_ ? inner : outer).stopReason());
  return ret;
}

void TrackMerger::addSecondTrackHits(std::vector<const TrackingRecHit *> &hits, const reco::Track &outer) const {
  size_t nHitsFirstTrack = hits.size();
  for (auto it = outer.recHitsBegin(), ed = outer.recHitsEnd(); it != ed; ++it) {
    const TrackingRecHit *hit = &**it;
    DetId id(hit->geographicalId());
    const auto lay = theTrkTopo->layer(id);
    bool shared = false;
    bool valid = hit->isValid();
    PRINT << "   subdet " << id.subdetId() << "  layer " << theTrkTopo->layer(id) << " valid " << valid
          << "   detid: " << id() << std::endl;
    size_t iHit = 0;
    for (auto hit2 : hits) {
      ++iHit;
      if (iHit > nHitsFirstTrack)
        break;
      DetId id2 = hit2->geographicalId();
      if (id.subdetId() != id2.subdetId())
        continue;
      if (theTrkTopo->layer(id2) != lay)
        continue;
      if (hit->sharesInput(hit2, TrackingRecHit::all)) {
        PRINT << "        discared as duplicate of other hit" << id() << std::endl;
        shared = true;
        break;
      }
      if (hit2->isValid() && !valid) {
        PRINT << "        replacing old invalid hit on detid " << id2() << std::endl;
        hit = hit2;
        shared = true;
        break;
      }
      PRINT << "        discared as additional hit on layer that already contains hit with detid " << id() << std::endl;
      shared = true;
      break;
    }
    if (shared)
      continue;
    hits.push_back(hit);
  }
}

void TrackMerger::sortByHitPosition(const GlobalVector &v,
                                    const std::vector<const TrackingRecHit *> &hits,
                                    TrackCandidate::RecHitContainer &ownHits) const {
  // NEW sort, more accurate
  unsigned int nhits = hits.size();
  std::vector<TransientTrackingRecHit::RecHitPointer> ttrh(nhits);
  for (unsigned int i = 0; i < nhits; ++i)
    ttrh[i] = theBuilder->build(hits[i]);
  std::sort(ttrh.begin(), ttrh.end(), GlobalMomentumSort(v));
  for (const auto &hit : ttrh)
    ownHits.push_back(*hit);
}

bool TrackMerger::MomentumSort::operator()(const TrackingRecHit *hit1, const TrackingRecHit *hit2) const {
  const GeomDet *det1 = geom_->idToDet(hit1->geographicalId());
  const GeomDet *det2 = geom_->idToDet(hit2->geographicalId());
  GlobalPoint p1 = det1->toGlobal(LocalPoint(0, 0, 0));
  GlobalPoint p2 = det2->toGlobal(LocalPoint(0, 0, 0));
  return (p2 - p1).dot(dir_) > 0;
}

bool TrackMerger::GlobalMomentumSort::operator()(const TransientTrackingRecHit::RecHitPointer &hit1,
                                                 const TransientTrackingRecHit::RecHitPointer &hit2) const {
  GlobalPoint p1 = hit1->isValid() ? hit1->globalPosition() : hit1->det()->position();
  GlobalPoint p2 = hit2->isValid() ? hit2->globalPosition() : hit2->det()->position();
  return (p2 - p1).dot(dir_) > 0;
}
