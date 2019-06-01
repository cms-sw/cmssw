#ifndef DetLayers_MuRingForwardLayer_H
#define DetLayers_MuRingForwardLayer_H

/** \class MuRingForwardLayer
 *  A plane composed of disks (MuRingForwardDisk). Represents forward muon CSC/RPC stations.
 *
 *  \author N. Amapane - INFN Torino
 *
 */

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"

class ForwardDetRing;
class ForwardDetRingBuilder;
class GeomDet;

class MuRingForwardLayer : public RingedForwardLayer {
public:
  /// Constructor, takes ownership of pointers
  MuRingForwardLayer(const std::vector<const ForwardDetRing*>& rings);

  ~MuRingForwardLayer() override;

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

  /// Return the vector of rings.
  virtual const std::vector<const ForwardDetRing*>& rings() const { return theRings; }

private:
  std::vector<const ForwardDetRing*> theRings;
  std::vector<const GeometricSearchDet*> theComponents;  // duplication of the above
  std::vector<const GeomDet*> theBasicComps;             // All chambers
  BaseBinFinder<double>* theBinFinder;
  bool isOverlapping;
};
#endif
