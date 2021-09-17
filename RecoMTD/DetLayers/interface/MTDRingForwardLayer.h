#ifndef RecoMTD_DetLayers_MTDRingForwardLayer_H
#define RecoMTD_DetLayers_MTDRingForwardLayer_H

/** \class MTDRingForwardLayer
 *  A plane composed of disks (MTDRingForwardDisk). Represents ETL.
 *
 *  \author L. Gray - FNAL
 *
 */

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"

class ForwardDetRing;
class ForwardDetRingBuilder;
class GeomDet;

class MTDRingForwardLayer : public RingedForwardLayer {
public:
  /// Constructor, takes ownership of pointers
  MTDRingForwardLayer(const std::vector<const ForwardDetRing*>& rings);

  ~MTDRingForwardLayer() override;

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
