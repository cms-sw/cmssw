#ifndef DetLayers_MTDDetRing_H
#define DetLayers_MTDDetRing_H

/** \class MTDDetRing
 *  A ring of periodic, possibly overlapping vertical detectors.
 *  Designed for the endcap timing layer.
 *
 *  \author L. Gray - FNAL
 */

#include "TrackingTools/DetLayers/interface/ForwardDetRingOneZ.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

class GeomDet;

class MTDDetRing : public ForwardDetRingOneZ {
 public:

  /// Construct from iterators on GeomDet*
  MTDDetRing(std::vector<const GeomDet*>::const_iterator first,
	    std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of GeomDet*
  MTDDetRing(const std::vector<const GeomDet*>& dets);

  ~MTDDetRing() override;


  // GeometricSearchDet interface

  const std::vector<const GeometricSearchDet*>& components() const override;

  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator& prop, 
	      const MeasurementEstimator& est) const override;

  std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const override;

  std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const override;


 private:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;
  BinFinderType theBinFinder;

  void init();

};
#endif

