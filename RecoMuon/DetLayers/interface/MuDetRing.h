#ifndef DetLayers_MuDetRing_H
#define DetLayers_MuDetRing_H

/** \class MuDetRing
 *  A ring of periodic, possibly overlapping vertical detectors.
 *  Designed for forward muon CSC/RPC chambers.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "TrackingTools/DetLayers/interface/ForwardDetRingOneZ.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

class GeomDet;

class MuDetRing : public ForwardDetRingOneZ {
public:
  /// Construct from iterators on GeomDet*
  MuDetRing(std::vector<const GeomDet*>::const_iterator first, std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of GeomDet*
  MuDetRing(const std::vector<const GeomDet*>& dets);

  ~MuDetRing() override;

  // GeometricSearchDet interface

  const std::vector<const GeometricSearchDet*>& components() const override;

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface& ts,
                                                       const Propagator& prop,
                                                       const MeasurementEstimator& est) const override;

  std::vector<DetWithState> compatibleDets(const TrajectoryStateOnSurface& startingState,
                                           const Propagator& prop,
                                           const MeasurementEstimator& est) const override;

  std::vector<DetGroup> groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                              const Propagator& prop,
                                              const MeasurementEstimator& est) const override;

private:
  typedef PeriodicBinFinderInPhi<float> BinFinderType;
  BinFinderType theBinFinder;

  void init();
};
#endif
