#ifndef DetLayers_MuDetRod_H
#define DetLayers_MuDetRod_H

/** \class MuDetRod
 *  A rod of aligned equal-sized non-overlapping detectors.  
 *  Designed for barrel muon DT/RPC chambers.
 *
 *  \author N. Amapane - INFN Torino
 *
 */

#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"
#include "Utilities/BinningTools/interface/GenericBinFinderInZ.h"
class GeomDet;

class MuDetRod : public DetRodOneR {
 public:

  /// Construct from iterators on GeomDet*
  MuDetRod(std::vector<const GeomDet*>::const_iterator first,
	   std::vector<const GeomDet*>::const_iterator last);

  /// Construct from a std::vector of GeomDet*
  MuDetRod(const std::vector<const GeomDet*>& dets);

  /// Destructor
  ~MuDetRod() override;


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
  //typedef PeriodicBinFinderInZ<float>   BinFinderType;
  typedef GenericBinFinderInZ<float, GeomDet> BinFinderType;
  BinFinderType theBinFinder;

  void init();

};

#endif
