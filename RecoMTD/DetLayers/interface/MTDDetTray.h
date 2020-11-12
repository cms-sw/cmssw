#ifndef RecoMTD_DetLayers_MTDDetTray_H
#define RecoMTD_DetLayers_MTDDetTray_H

/** \class MTDDetTray
 *  A tray of aligned equal-sized non-overlapping detectors.  
 *  Designed for barrel timing layer.
 *
 *  \author L. Gray - FNAL
 *
 */

#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "Utilities/BinningTools/interface/GenericBinFinderInZ.h"
class GeomDet;

class MTDDetTray : public DetRodOneR {
public:
  /// Construct from iterators on GeomDet*
  MTDDetTray(std::vector<const GeomDet*>::const_iterator first, std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a std::vector of GeomDet*
  MTDDetTray(const std::vector<const GeomDet*>& dets);

  /// Destructor
  ~MTDDetTray() override;

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
  typedef GenericBinFinderInZ<float, GeomDet> BinFinderType;
  BinFinderType theBinFinder;

  void init();
};

#endif
