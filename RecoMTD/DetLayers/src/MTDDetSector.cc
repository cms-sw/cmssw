//#define EDM_ML_DEBUG

#include "RecoMTD/DetLayers/interface/MTDDetSector.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "MTDDiskSectorBuilderFromDet.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

MTDDetSector::MTDDetSector(vector<const GeomDet*>::const_iterator first,
                           vector<const GeomDet*>::const_iterator last,
                           const MTDTopology& topo)
    : GeometricSearchDet(false), theDets(first, last), topo_(&topo) {
  init();
}

MTDDetSector::MTDDetSector(const vector<const GeomDet*>& vdets, const MTDTopology& topo)
    : GeometricSearchDet(false), theDets(vdets), topo_(&topo) {
  init();
}

void MTDDetSector::init() {
  // Add here the sector build based on a collection of GeomDets, mimic what done in ForwardDetRingOneZ
  // using the code from tracker BladeShapeBuilderFromDet
  // simple initial version, no sorting for the time being
  setDisk(MTDDiskSectorBuilderFromDet()(theDets));
}

const vector<const GeometricSearchDet*>& MTDDetSector::components() const {
  // FIXME dummy impl.
  edm::LogError("MTDDetLayers") << "temporary dummy implementation of MTDDetSector::components()!!";
  static const vector<const GeometricSearchDet*> result;
  return result;
}

pair<bool, TrajectoryStateOnSurface> MTDDetSector::compatible(const TrajectoryStateOnSurface& ts,
                                                              const Propagator& prop,
                                                              const MeasurementEstimator& est) const {
  TrajectoryStateOnSurface ms = prop.propagate(ts, specificSurface());

#ifdef EDM_ML_DEBUG
  LogTrace("MTDDetLayers") << "MTDDetSector::compatible, sector: \n"
                           << (*this) << "\n  TS at Z,R,phi: " << std::fixed << std::setw(14) << ts.globalPosition().z()
                           << " , " << std::setw(14) << ts.globalPosition().perp() << " , " << std::setw(14)
                           << ts.globalPosition().phi();
  if (ms.isValid()) {
    LogTrace("MTDDetLayers") << " DEST at Z,R,phi: " << std::fixed << std::setw(14) << ms.globalPosition().z() << " , "
                             << std::setw(14) << ms.globalPosition().perp() << " , " << std::setw(14)
                             << ms.globalPosition().phi() << " local Z: " << std::setw(14) << ms.localPosition().z();
  } else {
    LogTrace("MTDDetLayers") << " DEST: not valid";
  }
#endif

  return make_pair(ms.isValid() and est.estimate(ms, specificSurface()) != 0, ms);
}

vector<GeometricSearchDet::DetWithState> MTDDetSector::compatibleDets(const TrajectoryStateOnSurface& startingState,
                                                                      const Propagator& prop,
                                                                      const MeasurementEstimator& est) const {
  LogTrace("MTDDetLayers") << "MTDDetSector::compatibleDets, sector: \n"
                           << (*this) << "\n  TS at Z,R,phi: " << std::fixed << std::setw(14)
                           << startingState.globalPosition().z() << " , " << std::setw(14)
                           << startingState.globalPosition().perp() << " , " << std::setw(14)
                           << startingState.globalPosition().phi();

  vector<DetWithState> result;

  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat = compatible(startingState, prop, est);
  if (!compat.first) {
    LogTrace("MTDDetLayers") << "    MTDDetSector::compatibleDets: not compatible"
                             << "    (should not have been selected!)";
    return result;
  }

  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();

  LogTrace("MTDDetLayers") << "Starting position: " << startPos << " starting p/pT: " << tsos.globalMomentum().mag()
                           << " / " << tsos.globalMomentum().perp();

  // determine distance of det center from extrapolation on the surface, sort dets accordingly

  size_t idetMin = basicComponents().size();
  double dist2Min = std::numeric_limits<double>::max();
  std::vector<std::pair<double, size_t> > tmpDets;
  tmpDets.reserve(basicComponents().size());

  for (size_t idet = 0; idet < basicComponents().size(); idet++) {
    double dist2 = (startPos - theDets[idet]->position()).mag2();
    tmpDets.emplace_back(dist2, idet);
    if (dist2 < dist2Min) {
      dist2Min = dist2;
      idetMin = idet;
    }
  }

  //look for the compatibledets considering each line of the sector

  if (add(idetMin, result, tsos, prop, est)) {
    compatibleDetsLine(idetMin, result, tsos, prop, est);

    for (int iside = -1; iside <= 1; iside += 2) {
      bool isCompatible(true);
      size_t idetNew(idetMin);
      size_t closest = theDets.size();

      while (isCompatible) {
        idetNew = vshift(theDets[idetNew]->geographicalId().rawId(), iside, closest);
        if (idetNew >= theDets.size()) {
          if (closest < theDets.size()) {
            idetNew = closest;
          } else {
            break;
          }
        }
        isCompatible = add(idetNew, result, tsos, prop, est);
        if (isCompatible) {
          compatibleDetsLine(idetNew, result, tsos, prop, est);
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  if (result.empty()) {
    LogTrace("MTDDetLayers") << "MTDDetSector::compatibleDets, closest not compatible!";
  } else {
    LogTrace("MTDDetLayers") << "MTDDetSector::compatibleDets, found " << result.size() << " compatible dets";
  }
#endif

  return result;
}

void MTDDetSector::compatibleDetsV(const TrajectoryStateOnSurface&,
                                   const Propagator&,
                                   const MeasurementEstimator&,
                                   std::vector<DetWithState>&) const {
  edm::LogError("MTDDetLayers") << "At the moment not a real implementation";
}

vector<DetGroup> MTDDetSector::groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                                     const Propagator& prop,
                                                     const MeasurementEstimator& est) const {
  // FIXME should be implemented to allow returning  overlapping chambers
  // as separate groups!
  edm::LogInfo("MTDDetLayers") << "dummy implementation of MTDDetSector::groupedCompatibleDets()";
  vector<DetGroup> result;
  return result;
}

bool MTDDetSector::add(size_t idet,
                       vector<DetWithState>& result,
                       const TrajectoryStateOnSurface& tsos,
                       const Propagator& prop,
                       const MeasurementEstimator& est) const {
  pair<bool, TrajectoryStateOnSurface> compat = theCompatibilityChecker.isCompatible(theDets[idet], tsos, prop, est);

  if (compat.first) {
    result.push_back(DetWithState(theDets[idet], compat.second));
    LogTrace("MTDDetLayers") << "MTDDetSector::compatibleDets found compatible det idetMin " << idet
                             << " detId = " << theDets[idet]->geographicalId().rawId() << " at "
                             << theDets[idet]->position()
                             << " dist = " << std::sqrt((tsos.globalPosition() - theDets[idet]->position()).mag2());
  }

  return compat.first;
}

std::ostream& operator<<(std::ostream& os, const MTDDetSector& id) {
  os << " MTDDetSector at " << std::fixed << id.specificSurface().position() << std::endl
     << " L/W/T   : " << std::setw(14) << id.specificSurface().bounds().length() << " / " << std::setw(14)
     << id.specificSurface().bounds().width() << " / " << std::setw(14) << id.specificSurface().bounds().thickness()
     << std::endl
     << " rmin    : " << std::setw(14) << id.specificSurface().innerRadius() << std::endl
     << " rmax    : " << std::setw(14) << id.specificSurface().outerRadius() << std::endl
     << " phi ref : " << std::setw(14) << id.specificSurface().position().phi() << std::endl
     << " phi w/2 : " << std::setw(14) << id.specificSurface().phiHalfExtension() << std::endl;
  return os;
}

void MTDDetSector::compatibleDetsLine(const size_t idetMin,
                                      vector<DetWithState>& result,
                                      const TrajectoryStateOnSurface& tsos,
                                      const Propagator& prop,
                                      const MeasurementEstimator& est) const {
  for (int iside = -1; iside <= 1; iside += 2) {
    bool isCompatible(true);
    size_t idetNew(idetMin);

    while (isCompatible) {
      idetNew = hshift(theDets[idetNew]->geographicalId().rawId(), iside);
      if (idetNew >= theDets.size()) {
        break;
      }
      isCompatible = add(idetNew, result, tsos, prop, est);
    }
  }

  return;
}

size_t MTDDetSector::hshift(const uint32_t detid, const int horizontalShift) const {
  return topo_->hshiftETL(detid, horizontalShift);
}

size_t MTDDetSector::vshift(const uint32_t detid, const int verticalShift, size_t& closest) const {
  return topo_->vshiftETL(detid, verticalShift, closest);
}
