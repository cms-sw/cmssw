#ifndef RecoMTD_DetLayers_MTDSectorForwardDoubleLayer_H
#define RecoMTD_DetLayers_MTDSectorForwardDoubleLayer_H

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardLayer.h"

class MTDDetSector;
class GeomDet;

class MTDSectorForwardDoubleLayer : public ForwardDetLayer {
public:
  /// Constructor, takes ownership of pointers
  MTDSectorForwardDoubleLayer(const std::vector<const MTDDetSector*>& frontSectors,
                              const std::vector<const MTDDetSector*>& backSectors);

  ~MTDSectorForwardDoubleLayer() override {}

  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override { return theBasicComponents; }

  const std::vector<const GeometricSearchDet*>& components() const override { return theComponents; }

  bool isInsideOut(const TrajectoryStateOnSurface& tsos) const;

  // tries closest layer first
  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface&,
                                                       const Propagator&,
                                                       const MeasurementEstimator&) const override;

  std::vector<DetWithState> compatibleDets(const TrajectoryStateOnSurface& startingState,
                                           const Propagator& prop,
                                           const MeasurementEstimator& est) const override;

  std::vector<DetGroup> groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                              const Propagator& prop,
                                              const MeasurementEstimator& est) const override;

  // DetLayer interface
  SubDetector subDetector() const override { return theBackLayer.subDetector(); }

  // Extension of the interface

  /// Return the vector of sectors.
  virtual const std::vector<const MTDDetSector*>& sectors() const { return theSectors; }

  bool isCrack(const GlobalPoint& gp) const;

  const MTDSectorForwardLayer* frontLayer() const { return &theFrontLayer; }
  const MTDSectorForwardLayer* backLayer() const { return &theBackLayer; }

  void selfTest() const;

protected:
  BoundDisk* computeSurface() override;

private:
  MTDSectorForwardLayer theFrontLayer;
  MTDSectorForwardLayer theBackLayer;
  std::vector<const MTDDetSector*> theSectors;
  std::vector<const GeometricSearchDet*> theComponents;  // duplication of the above
  std::vector<const GeomDet*> theBasicComponents;        // All chambers
};
#endif
