#ifndef RecoMTD_DetLayers_MTDDetSector_H
#define RecoMTD_DetLayers_MTDDetSector_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"

#include <ostream>

class GeomDet;

class MTDDetSector : public GeometricSearchDet {
public:
  using GeometricSearchDet::GeometricSearchDet;

  /// Construct from iterators on GeomDet*
  MTDDetSector(std::vector<const GeomDet*>::const_iterator first, std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of GeomDet*
  MTDDetSector(const std::vector<const GeomDet*>& dets);

  ~MTDDetSector() override;

  // GeometricSearchDet structure

  const std::vector<const GeomDet*>& basicComponents() const override { return theDets; }

  const BoundSurface& surface() const final { return *theDiskS; }

  const std::vector<const GeometricSearchDet*>& components() const override;

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface& ts,
                                                       const Propagator& prop,
                                                       const MeasurementEstimator& est) const override;

  std::vector<DetWithState> compatibleDets(const TrajectoryStateOnSurface& startingState,
                                           const Propagator& prop,
                                           const MeasurementEstimator& est) const override;

  void compatibleDetsV(const TrajectoryStateOnSurface& startingState,
                       const Propagator& prop,
                       const MeasurementEstimator& est,
                       std::vector<DetWithState>& result) const override;

  std::vector<DetGroup> groupedCompatibleDets(const TrajectoryStateOnSurface& startingState,
                                              const Propagator& prop,
                                              const MeasurementEstimator& est) const override;

  // GeometricSearchDet extension

  const BoundDiskSector& specificSurface() const { return *theDiskS; }

protected:
  void setDisk(BoundDiskSector* diskS) { theDiskS = diskS; }

  bool add(int idet,
           std::vector<DetWithState>& result,
           const TrajectoryStateOnSurface& tsos,
           const Propagator& prop,
           const MeasurementEstimator& est) const;

private:
  ReferenceCountingPointer<BoundDiskSector> theDiskS;
  std::vector<const GeomDet*> theDets;

  void init();
};

std::ostream& operator<<(std::ostream&, const MTDDetSector&);

#endif
