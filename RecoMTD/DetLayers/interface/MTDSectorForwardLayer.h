#ifndef RecoMTD_DetLayers_MTDSectorForwardLayer_H
#define RecoMTD_DetLayers_MTDSectorForwardLayer_H

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

class MTDDetSector;
class GeomDet;

class MTDSectorForwardLayer : public ForwardDetLayer {
public:
  /// Constructor, takes ownership of pointers
  MTDSectorForwardLayer(const std::vector<const MTDDetSector*>& sectors);

  ~MTDSectorForwardLayer() override;

  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override { return theBasicComps; }

  const std::vector<const GeometricSearchDet*>& components() const override;

  std::vector<DetWithState> compatibleDets(const TrajectoryStateOnSurface& startingState,
                                           const Propagator& prop,
                                           const MeasurementEstimator& est) const override;

  std::vector<DetGroup> groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                              const Propagator& prop,
                                              const MeasurementEstimator& est) const override;

  // DetLayer interface
  SubDetector subDetector() const override;

  // Extension of the interface

  /// Return the vector of sectors
  virtual const std::vector<const MTDDetSector*>& sectors() const { return theSectors; }

private:
  std::vector<const MTDDetSector*> theSectors;
  std::vector<const GeometricSearchDet*> theComponents;  // duplication of the above
  std::vector<const GeomDet*> theBasicComps;             // All chambers
};
#endif
