#include "RecoMTD/TimingTools/interface/TrackPathLength.h"

#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/trajectoryStateClosestToBeamLine.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace mtd {

  bool trackPathLength(const Trajectory& traj,
                       const TrajectoryStateClosestToBeamLine& tscbl,
                       const Propagator* propagator,
                       float& pathLength,
                       TrackSegments& trs) {
    pathLength = 0.f;

    bool validpropagation = true;
    float oldp = traj.measurements().begin()->updatedState().globalMomentum().mag();
    float pathlength1 = 0.f;
    float pathlength2 = 0.f;

    //add pathlength layer by layer
    for (auto it = traj.measurements().begin(); it != traj.measurements().end() - 1; ++it) {
      const auto& propresult = propagator->propagateWithPath(it->updatedState(), (it + 1)->updatedState().surface());
      float layerpathlength = std::abs(propresult.second);
      if (layerpathlength == 0.f) {
        validpropagation = false;
      }
      pathlength1 += layerpathlength;

      // sigma(p) from curvilinear error (on q/p)
      float sigma_p = sqrt((it + 1)->updatedState().curvilinearError().matrix()(0, 0)) *
                      (it + 1)->updatedState().globalMomentum().mag2();

      trs.addSegment(layerpathlength, (it + 1)->updatedState().globalMomentum().mag2(), sigma_p);

      LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_i " << std::fixed
                                       << std::setw(14) << it->updatedState().globalPosition().perp() << " z_i "
                                       << std::fixed << std::setw(14) << it->updatedState().globalPosition().z()
                                       << " R_e " << std::fixed << std::setw(14)
                                       << (it + 1)->updatedState().globalPosition().perp() << " z_e " << std::fixed
                                       << std::setw(14) << (it + 1)->updatedState().globalPosition().z() << " p "
                                       << std::fixed << std::setw(14) << (it + 1)->updatedState().globalMomentum().mag()
                                       << " dp " << std::fixed << std::setw(14)
                                       << (it + 1)->updatedState().globalMomentum().mag() - oldp;
      oldp = (it + 1)->updatedState().globalMomentum().mag();
    }

    //add distance from bs to first measurement
    auto const& tscblPCA = tscbl.trackStateAtPCA();
    auto const& aSurface = traj.direction() == alongMomentum ? traj.firstMeasurement().updatedState().surface()
                                                             : traj.lastMeasurement().updatedState().surface();
    pathlength2 = propagator->propagateWithPath(tscblPCA, aSurface).second;
    if (pathlength2 == 0.f) {
      validpropagation = false;
    }
    pathLength = pathlength1 + pathlength2;

    float sigma_p = sqrt(tscblPCA.curvilinearError().matrix()(0, 0)) * tscblPCA.momentum().mag2();

    trs.addSegment(pathlength2, tscblPCA.momentum().mag2(), sigma_p);

    LogTrace("TrackExtenderWithMTD") << "TSOS " << std::fixed << std::setw(4) << trs.size() << " R_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().perp() << " z_e " << std::fixed
                                     << std::setw(14) << tscblPCA.position().z() << " p " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() << " dp " << std::fixed << std::setw(14)
                                     << tscblPCA.momentum().mag() - oldp << " sigma_p = " << std::fixed << std::setw(14)
                                     << sigma_p << " sigma_p/p = " << std::fixed << std::setw(14)
                                     << sigma_p / tscblPCA.momentum().mag() * 100 << " %";

    return validpropagation;
  }

  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* propagator,
                       float& pathLength,
                       TrackSegments& trs) {
    pathLength = 0.f;

    TrajectoryStateClosestToBeamLine tscbl;
    if (!trajectoryStateClosestToBeamLine(traj, bs, propagator, tscbl))
      return false;

    return trackPathLength(traj, tscbl, propagator, pathLength, trs);
  }

}  // namespace mtd
