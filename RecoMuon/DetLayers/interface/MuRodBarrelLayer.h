#ifndef DetLayers_MuRodBarrelLayer_H
#define DetLayers_MuRodBarrelLayer_H

/** \class MuRodBarrelLayer
 *  A cylinder composed of rods. Represents barrel muon DT/RPC stations.
 *
 *  \author N. Amapane - INFN Torino
 *
 */
#include "TrackingTools/DetLayers/interface/RodBarrelLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"

class DetRod;
class DetRodBuilder;
class GeomDet;

class MuRodBarrelLayer : public RodBarrelLayer {
public:
  /// Constructor, takes ownership of pointers
  MuRodBarrelLayer(std::vector<const DetRod*>& rods);

  ~MuRodBarrelLayer() override;

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

  /// Return the vector of rods.
  virtual const std::vector<const DetRod*>& rods() const { return theRods; }

private:
  float xError(const TrajectoryStateOnSurface& tsos, const MeasurementEstimator& est) const;

  std::vector<const DetRod*> theRods;
  std::vector<const GeometricSearchDet*> theComponents;  // duplication of the above
  std::vector<const GeomDet*> theBasicComps;             // All chambers
  BaseBinFinder<double>* theBinFinder;
  bool isOverlapping;
};

#endif
