#include "TkGluedMeasurementDet.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripMatchedRecHit.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "NonPropagatingDetMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "RecHitPropagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>
#include <iostream>
#include <memory>
#include <typeinfo>

namespace {
  inline std::pair<LocalPoint, LocalError> projectedPos(const TrackingRecHit& hit,
                                                        const GeomDet& det,
                                                        const GlobalVector& gdir,
                                                        const StripClusterParameterEstimator* cpe) {
    const BoundPlane& gluedPlane = det.surface();
    const BoundPlane& hitPlane = hit.det()->surface();

    // check if the planes are parallel
    //const float epsilon = 1.e-7; // corresponds to about 0.3 miliradian but cannot be reduced
    // because of float precision

    //if (fabs(gluedPlane.normalVector().dot( hitPlane.normalVector())) < 1-epsilon) {
    //       std::cout << "TkGluedMeasurementDet plane not parallel to DetUnit plane: dot product is "
    //     << gluedPlane.normalVector().dot( hitPlane.normalVector()) << endl;
    // FIXME: throw the appropriate exception here...
    //throw MeasurementDetException("TkGluedMeasurementDet plane not parallel to DetUnit plane");
    //}

    double delta = gluedPlane.localZ(hitPlane.position());
    LocalVector ldir = gluedPlane.toLocal(gdir);
    LocalPoint lhitPos = gluedPlane.toLocal(hit.globalPosition());
    LocalPoint projectedHitPos = lhitPos - ldir * delta / ldir.z();

    LocalVector hitXAxis = gluedPlane.toLocal(hitPlane.toGlobal(LocalVector(1, 0, 0)));
    LocalError hitErr = hit.localPositionError();
    if (gluedPlane.normalVector().dot(hitPlane.normalVector()) < 0) {
      // the two planes are inverted, and the correlation element must change sign
      hitErr = LocalError(hitErr.xx(), -hitErr.xy(), hitErr.yy());
    }
    LocalError rotatedError = hitErr.rotate(hitXAxis.x(), hitXAxis.y());

    return std::make_pair(projectedHitPos, rotatedError);
  }
}  // namespace

// #include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"

using namespace std;

TkGluedMeasurementDet::TkGluedMeasurementDet(const GluedGeomDet* gdet,
                                             const SiStripRecHitMatcher* matcher,
                                             const StripClusterParameterEstimator* cpe)
    : MeasurementDet(gdet),
      theMatcher(matcher),
      theCPE(cpe),
      theMonoDet(nullptr),
      theStereoDet(nullptr),
      theTopology(nullptr) {}

void TkGluedMeasurementDet::init(const MeasurementDet* monoDet,
                                 const MeasurementDet* stereoDet,
                                 const TrackerTopology* tTopo) {
  theMonoDet = dynamic_cast<const TkStripMeasurementDet*>(monoDet);
  theStereoDet = dynamic_cast<const TkStripMeasurementDet*>(stereoDet);
  theTopology = tTopo;

  if ((theMonoDet == nullptr) || (theStereoDet == nullptr)) {
    throw MeasurementDetException(
        "TkGluedMeasurementDet ERROR: Trying to glue a det which is not a TkStripMeasurementDet");
  }
}

TkGluedMeasurementDet::RecHitContainer TkGluedMeasurementDet::recHits(const TrajectoryStateOnSurface& ts,
                                                                      const MeasurementTrackerEvent& data) const {
  RecHitContainer result;
  HitCollectorForRecHits collector(&fastGeomDet(), theMatcher, theCPE, result);
  collectRecHits(ts, data, collector);
  return result;
}

// simple hits
bool TkGluedMeasurementDet::recHits(SimpleHitContainer& result,
                                    const TrajectoryStateOnSurface& stateOnThisDet,
                                    const MeasurementEstimator& est,
                                    const MeasurementTrackerEvent& data) const {
  if UNLIKELY ((!theMonoDet->isActive(data)) && (!theStereoDet->isActive(data)))
    return false;
  auto oldSize = result.size();
  HitCollectorForSimpleHits collector(&fastGeomDet(), theMatcher, theCPE, stateOnThisDet, est, result);
  collectRecHits(stateOnThisDet, data, collector);

  return result.size() > oldSize;
}

bool TkGluedMeasurementDet::measurements(const TrajectoryStateOnSurface& stateOnThisDet,
                                         const MeasurementEstimator& est,
                                         const MeasurementTrackerEvent& data,
                                         TempMeasurements& result) const {
  if UNLIKELY ((!theMonoDet->isActive(data)) && (!theStereoDet->isActive(data))) {
    //     LogDebug("TkStripMeasurementDet") << " DetID " << geomDet().geographicalId().rawId() << " (glued) fully inactive";
    result.add(theInactiveHit, 0.F);
    return true;
  }

  auto oldSize = result.size();

  HitCollectorForFastMeasurements collector(&fastGeomDet(), theMatcher, theCPE, stateOnThisDet, est, result);
  collectRecHits(stateOnThisDet, data, collector);

  if (result.size() > oldSize)
    return true;

  auto id = geomDet().geographicalId().subdetId() - 3;
  auto l = theTopology->tobLayer(geomDet().geographicalId());
  bool killHIP = (1 == l) && (2 == id);  //TOB1
  killHIP &= stateOnThisDet.globalMomentum().perp2() > est.minPt2ForHitRecoveryInGluedDet();
  if (killHIP) {
    result.add(theInactiveHit, 0.F);
    return true;
  }

  //LogDebug("TkStripMeasurementDet") << "No hit found on TkGlued. Testing strips...  ";
  const BoundPlane& gluedPlane = geomDet().surface();
  if (  // sorry for the big IF, but I want to exploit short-circuiting of logic
      stateOnThisDet.hasError() &&
      (/* do this only if the state has uncertainties, otherwise it will throw 
					 (states without uncertainties are passed to this code from seeding */
       (theMonoDet->isActive(data) &&
        (theMonoDet->hasAllGoodChannels() || testStrips(stateOnThisDet, gluedPlane, *theMonoDet))) /*Mono OK*/
       &&                                                                                          // was ||
       (theStereoDet->isActive(data) &&
        (theStereoDet->hasAllGoodChannels() || testStrips(stateOnThisDet, gluedPlane, *theStereoDet))) /*Stereo OK*/
       ) /* State has errors */
  ) {
    result.add(theMissingHit, 0.F);
    return false;
  }
  result.add(theInactiveHit, 0.F);
  return true;
}

struct take_address {
  template <typename T>
  const T* operator()(const T& val) const {
    return &val;
  }
};

#ifdef DOUBLE_MATCH
template <typename Collector>
void TkGluedMeasurementDet::collectRecHits(const TrajectoryStateOnSurface& ts,
                                           const MeasurementTrackerEvent& data,
                                           Collector& collector) const {
  doubleMatch(ts, data, collector);
}
#else
template <typename Collector>
void TkGluedMeasurementDet::collectRecHits(const TrajectoryStateOnSurface& ts,
                                           const MeasurementTrackerEvent& data,
                                           Collector& collector) const {
  //------ WARNING: here ts is used as it is on the mono/stereo surface.
  //-----           A further propagation is necessary.
  //-----           To limit the problem, the SimpleCPE should be used
  RecHitContainer monoHits = theMonoDet->recHits(ts, data);
  GlobalVector glbDir = (ts.isValid() ? ts.globalParameters().momentum() : position() - GlobalPoint(0, 0, 0));

  //edm::LogWarning("TkGluedMeasurementDet::recHits") << "Query-for-detid-" << theGeomDet->geographicalId().rawId();

  //checkProjection(ts, monoHits, stereoHits);

  if (monoHits.empty()) {
    // make stereo TTRHs and project them
    projectOnGluedDet(collector, theStereoDet->recHits(ts, data), glbDir);
  } else {
    // collect simple stereo hits
    std::vector<SiStripRecHit2D> simpleSteroHitsByValue;
    theStereoDet->simpleRecHits(ts, data, simpleSteroHitsByValue);

    if (simpleSteroHitsByValue.empty()) {
      projectOnGluedDet(collector, monoHits, glbDir);
    } else {
      LocalVector tkDir = (ts.isValid() ? ts.localDirection() : surface().toLocal(position() - GlobalPoint(0, 0, 0)));
      SiStripRecHitMatcher::SimpleHitCollection vsStereoHits;
      vsStereoHits.resize(simpleSteroHitsByValue.size());
      std::transform(
          simpleSteroHitsByValue.begin(), simpleSteroHitsByValue.end(), vsStereoHits.begin(), take_address());

      // convert mono hits to type expected by matcher
      for (RecHitContainer::const_iterator monoHit = monoHits.begin(); monoHit != monoHits.end(); ++monoHit) {
        const TrackingRecHit* tkhit = (**monoHit).hit();
        const SiStripRecHit2D* verySpecificMonoHit = reinterpret_cast<const SiStripRecHit2D*>(tkhit);
        theMatcher->match(verySpecificMonoHit,
                          vsStereoHits.begin(),
                          vsStereoHits.end(),
                          collector.collector(),
                          &specificGeomDet(),
                          tkDir);

        if (collector.hasNewMatchedHits()) {
          collector.clearNewMatchedHitsFlag();
        } else {
          collector.addProjected(**monoHit, glbDir);
        }
      }  // loop on mono hit
    }
    //GIO// std::cerr << "TkGluedMeasurementDet hits " << monoHits.size() << "/" << stereoHits.size() << " => " << result.size() << std::endl;
  }
}
#endif

#include <cstdint>
#ifdef VI_STAT
#include <cstdio>
#endif
namespace {
  struct Stat {
#ifdef VI_STAT
    double totCall = 0;
    double totMono = 0;
    double totStereo = 0;
    double totComb = 0;
    double totMatched = 0;
    double filtMono = 0;
    double filtStereo = 0;
    double filtComb = 0;
    double matchT = 0;
    double matchF = 0;
    double singleF = 0;
    double zeroM = 0;
    double zeroS = 0;

    void match(uint64_t t) {
      if (t != 0)
        ++matchT;
      totMatched += t;
    }
    void operator()(uint64_t m, uint64_t s, uint64_t fm, uint64_t fs) {
      ++totCall;
      totMono += m;
      totStereo += s;
      totComb += m * s;
      filtMono += fm;
      filtStereo += fs;
      filtComb += fm * fs;
      if (fm == 0)
        ++zeroM;
      if (fs == 0)
        ++zeroS;
      if (fm != 0 && fs != 0)
        ++matchF;
      if (fm != 0 || fs != 0)
        ++singleF;
    }
    ~Stat() {
      if (totCall > 0)
        printf("Matches:%d/%d/%d/%d/%d/%d : %f/%f/%f/%f/%f/%f/%f\n",
               int(totCall),
               int(matchF),
               int(singleF - matchF),
               int(matchT),
               int(zeroM),
               int(zeroS),
               totMono / totCall,
               totStereo / totCall,
               totComb / totCall,
               totMatched / matchT,
               filtMono / totCall,
               filtStereo / totCall,
               filtComb / matchF);
    }
#else
    Stat() {}
    void match(uint64_t) const {}
    void operator()(uint64_t, uint64_t, uint64_t, uint64_t) const {}
#endif
  };
#ifndef VI_STAT
  const
#endif
      Stat stat;
}  // namespace

TkGluedMeasurementDet::RecHitContainer TkGluedMeasurementDet::projectOnGluedDet(
    const RecHitContainer& hits, const TrajectoryStateOnSurface& ts) const {
  RecHitContainer result;
  for (auto const& hit : hits) {
    auto&& vl = projectedPos(*hit, fastGeomDet(), ts.globalParameters().momentum(), theCPE);
    auto&& phit = std::make_shared<ProjectedSiStripRecHit2D>(
        vl.first, vl.second, fastGeomDet(), static_cast<SiStripRecHit2D const&>(*hit));
    result.push_back(std::move(phit));
  }
  return result;
}

template <typename Collector>
void TkGluedMeasurementDet::projectOnGluedDet(Collector& collector,
                                              const RecHitContainer& hits,
                                              const GlobalVector& gdir) const {
  for (RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ihit++) {
    collector.addProjected(**ihit, gdir);
  }
}

TkGluedMeasurementDet::RecHitContainer TkGluedMeasurementDet::projectOnGluedDet(
    std::vector<SiStripRecHit2D> const& hits, const TrajectoryStateOnSurface& ts) const {
  RecHitContainer result;
  for (auto const& hit : hits) {
    auto&& vl = projectedPos(hit, fastGeomDet(), ts.globalParameters().momentum(), theCPE);
    auto&& phit = std::make_shared<ProjectedSiStripRecHit2D>(
        vl.first, vl.second, fastGeomDet(), static_cast<SiStripRecHit2D const&>(hit));
    result.push_back(std::move(phit));
  }
  return result;
}

template <typename Collector>
void TkGluedMeasurementDet::projectOnGluedDet(Collector& collector,
                                              std::vector<SiStripRecHit2D> const& hits,
                                              const GlobalVector& gdir) const {
  for (auto const& hit : hits)
    collector.addProjected(hit, gdir);
}

void TkGluedMeasurementDet::checkProjection(const TrajectoryStateOnSurface& ts,
                                            const RecHitContainer& monoHits,
                                            const RecHitContainer& stereoHits) const {
  for (RecHitContainer::const_iterator i = monoHits.begin(); i != monoHits.end(); ++i) {
    checkHitProjection(**i, ts, fastGeomDet());
  }
  for (RecHitContainer::const_iterator i = stereoHits.begin(); i != stereoHits.end(); ++i) {
    checkHitProjection(**i, ts, fastGeomDet());
  }
}

void TkGluedMeasurementDet::checkHitProjection(const TrackingRecHit& hit,
                                               const TrajectoryStateOnSurface& ts,
                                               const GeomDet& det) const {
  auto&& vl = projectedPos(hit, det, ts.globalParameters().momentum(), theCPE);
  ProjectedSiStripRecHit2D projectedHit(vl.first, vl.second, det, static_cast<SiStripRecHit2D const&>(hit));

  RecHitPropagator prop;
  TrajectoryStateOnSurface propState = prop.propagate(hit, det.surface(), ts);

  if ((projectedHit.localPosition() - propState.localPosition()).mag() > 0.0001f) {
    std::cout << "PROBLEM: projected and propagated hit positions differ by "
              << (projectedHit.localPosition() - propState.localPosition()).mag() << std::endl;
  }

  LocalError le1 = projectedHit.localPositionError();
  LocalError le2 = propState.localError().positionError();
  double eps = 1.e-5;
  double cutoff = 1.e-4;  // if element below cutoff, use absolute instead of relative accuracy
  double maxdiff = std::max(
      std::max(fabs(le1.xx() - le2.xx()) / (cutoff + le1.xx()), fabs(le1.xy() - le2.xy()) / (cutoff + fabs(le1.xy()))),
      fabs(le1.yy() - le2.yy()) / (cutoff + le1.xx()));
  if (maxdiff > eps) {
    std::cout << "PROBLEM: projected and propagated hit errors differ by " << maxdiff << std::endl;
  }
}

bool TkGluedMeasurementDet::testStrips(const TrajectoryStateOnSurface& tsos,
                                       const BoundPlane& gluedPlane,
                                       const TkStripMeasurementDet& mdet) const {
  // from TrackingRecHitProjector
  const GeomDet& det = mdet.fastGeomDet();
  const BoundPlane& stripPlane = det.surface();

  //LocalPoint glp = tsos.localPosition();
  LocalError err = tsos.localError().positionError();
  /*LogDebug("TkStripMeasurementDet") << 
    "Testing local pos glued: " << glp << 
    " local err glued: " << tsos.localError().positionError() << 
      " in? " << gluedPlane.bounds().inside(glp) <<
      " in(3s)? " << gluedPlane.bounds().inside(glp, err, 3.0f);*/

  GlobalVector gdir = tsos.globalParameters().momentum();

  LocalPoint slp = stripPlane.toLocal(tsos.globalPosition());
  LocalVector sld = stripPlane.toLocal(gdir);

  double delta = stripPlane.localZ(tsos.globalPosition());
  LocalPoint pos = slp - sld * delta / sld.z();

  // now the error
  LocalVector hitXAxis = stripPlane.toLocal(gluedPlane.toGlobal(LocalVector(1, 0, 0)));
  if (stripPlane.normalVector().dot(gluedPlane.normalVector()) < 0) {
    // the two planes are inverted, and the correlation element must change sign
    err = LocalError(err.xx(), -err.xy(), err.yy());
  }
  LocalError rotatedError = err.rotate(hitXAxis.x(), hitXAxis.y());

  /* // This is probably meaningless 
      LogDebug("TkStripMeasurementDet") << 
      "Testing local pos on strip (SLP): " << slp << 
      " in? :" << stripPlane.bounds().inside(slp) <<
      " in(3s)? :" << stripPlane.bounds().inside(slp, rotatedError, 3.0f);
      // but it helps to test bugs in the formula for POS */
  /*LogDebug("TkStripMeasurementDet") << 
     "Testing local pos strip: " << pos << 
     " in? " << stripPlane.bounds().inside(pos) <<
     " in(3s)? " << stripPlane.bounds().inside(pos, rotatedError, 3.0f);*/

  // now we need to convert to MeasurementFrame
  const StripTopology& topo = mdet.specificGeomDet().specificTopology();
  float utraj = topo.measurementPosition(pos).x();
  float uerr = std::sqrt(topo.measurementError(pos, rotatedError).uu());
  return mdet.testStrips(utraj, uerr);
}

TkGluedMeasurementDet::HitCollectorForRecHits::HitCollectorForRecHits(const GeomDet* geomDet,
                                                                      const SiStripRecHitMatcher* matcher,
                                                                      const StripClusterParameterEstimator* cpe,
                                                                      RecHitContainer& target)
    : geomDet_(geomDet),
      matcher_(matcher),
      cpe_(cpe),
      target_(target),
      collector_(std::bind(&HitCollectorForRecHits::add, std::ref(*this), std::placeholders::_1)),
      hasNewHits_(false) {}

TkGluedMeasurementDet::HitCollectorForSimpleHits::HitCollectorForSimpleHits(
    const GeomDet* geomDet,
    const SiStripRecHitMatcher* matcher,
    const StripClusterParameterEstimator* cpe,
    const TrajectoryStateOnSurface& stateOnThisDet,
    const MeasurementEstimator& est,
    SimpleHitContainer& target)
    : geomDet_(geomDet),
      matcher_(matcher),
      cpe_(cpe),
      stateOnThisDet_(stateOnThisDet),
      est_(est),
      target_(target),
      collector_(std::bind(&HitCollectorForSimpleHits::add, std::ref(*this), std::placeholders::_1)),
      hasNewHits_(false) {}

void TkGluedMeasurementDet::HitCollectorForRecHits::addProjected(const TrackingRecHit& hit, const GlobalVector& gdir) {
  auto&& vl = projectedPos(hit, *geomDet_, gdir, cpe_);
  auto&& phit = std::make_shared<ProjectedSiStripRecHit2D>(
      vl.first, vl.second, *geomDet_, static_cast<SiStripRecHit2D const&>(hit));
  target_.push_back(std::move(phit));
}

void TkGluedMeasurementDet::HitCollectorForSimpleHits::add(SiStripMatchedRecHit2D const& hit2d) {
  hasNewHits_ = true;  //FIXME: see also what happens moving this within testAndPush   // consistent with previous code
  if (!est_.preFilter(stateOnThisDet_,
                      ClusterFilterPayload(hit2d.geographicalId(), &hit2d.monoCluster(), &hit2d.stereoCluster())))
    return;
  hasNewHits_ = true;  //FIXME: see also what happens moving this within testAndPush

  std::pair<bool, double> diffEst = est_.estimate(stateOnThisDet_, hit2d);
  if (diffEst.first)
    target_.emplace_back(new SiStripMatchedRecHit2D(hit2d));  // fix to use move (really needed???)
}

void TkGluedMeasurementDet::HitCollectorForSimpleHits::addProjected(const TrackingRecHit& hit,
                                                                    const GlobalVector& gdir) {
  auto const& thit = reinterpret_cast<TrackerSingleRecHit const&>(hit);
  if (!est_.preFilter(stateOnThisDet_, ClusterFilterPayload(hit.geographicalId(), &thit.stripCluster())))
    return;

  // here we're ok with some extra casual new's and delete's
  auto&& vl = projectedPos(hit, *geomDet_, gdir, cpe_);
  std::unique_ptr<ProjectedSiStripRecHit2D> phit(
      new ProjectedSiStripRecHit2D(vl.first, vl.second, *geomDet_, static_cast<SiStripRecHit2D const&>(hit)));
  std::pair<bool, double> diffEst = est_.estimate(stateOnThisDet_, *phit);
  if (diffEst.first) {
    target_.emplace_back(phit.release());
  }
}

TkGluedMeasurementDet::HitCollectorForFastMeasurements::HitCollectorForFastMeasurements(
    const GeomDet* geomDet,
    const SiStripRecHitMatcher* matcher,
    const StripClusterParameterEstimator* cpe,
    const TrajectoryStateOnSurface& stateOnThisDet,
    const MeasurementEstimator& est,
    TempMeasurements& target)
    : geomDet_(geomDet),
      matcher_(matcher),
      cpe_(cpe),
      stateOnThisDet_(stateOnThisDet),
      est_(est),
      target_(target),
      collector_(std::bind(&HitCollectorForFastMeasurements::add, std::ref(*this), std::placeholders::_1)),
      hasNewHits_(false) {}

void TkGluedMeasurementDet::HitCollectorForFastMeasurements::add(SiStripMatchedRecHit2D const& hit2d) {
  //FIXME: see also what happens moving this within testAndPush  // consistent with previous code...
  hasNewHits_ = true;
  if (!est_.preFilter(stateOnThisDet_,
                      ClusterFilterPayload(hit2d.geographicalId(), &hit2d.monoCluster(), &hit2d.stereoCluster())))
    return;
  hasNewHits_ = true;  //FIXME: see also what happens moving this within testAndPush

  std::pair<bool, double> diffEst = est_.estimate(stateOnThisDet_, hit2d);
  if (diffEst.first)
    target_.add(hit2d.cloneSH(), diffEst.second);
}

void TkGluedMeasurementDet::HitCollectorForFastMeasurements::addProjected(const TrackingRecHit& hit,
                                                                          const GlobalVector& gdir) {
  auto const& thit = reinterpret_cast<TrackerSingleRecHit const&>(hit);
  if (!est_.preFilter(stateOnThisDet_, ClusterFilterPayload(hit.geographicalId(), &thit.stripCluster())))
    return;

  // here we're ok with some extra casual new's and delete's
  auto&& vl = projectedPos(hit, *geomDet_, gdir, cpe_);
  auto&& phit = std::make_shared<ProjectedSiStripRecHit2D>(
      vl.first, vl.second, *geomDet_, static_cast<SiStripRecHit2D const&>(hit));

  std::pair<bool, double> diffEst = est_.estimate(stateOnThisDet_, *phit);
  if (diffEst.first) {
    target_.add(phit, diffEst.second);
  }
}

#ifdef DOUBLE_MATCH
#include "doubleMatch.icc"
#endif
