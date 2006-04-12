#ifndef DetLayers_MuDetRod_H
#define DetLayers_MuDetRod_H

/** \class MuDetRod
 *  A rod of aligned equal-sized non-overlapping detectors.  
 *  Designed for barrel muon DT/RPC chambers.
 *
 *  $Date: 2004/03/08 16:02:01 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 *
 */

#include "TrackingTools/DetLayers/interface/DetRodOneR.h"
#include "TrackingTools/DetLayers/interface/PeriodicBinFinderInZ.h"

class GeomDet;

class MuDetRod : public DetRodOneR {
 public:

  /// Construct from iterators on GeomDet*
  MuDetRod(vector<const GeomDet*>::const_iterator first,
	   vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of GeomDet*
  MuDetRod(const vector<const GeomDet*>& dets);

  /// Destructor
  virtual ~MuDetRod();


  // GeometricSearchDet interface

  virtual vector<const GeometricSearchDet*> components() const;

  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator& prop, 
	      const MeasurementEstimator& est) const;

  virtual vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;

  virtual vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return false;}

 private:
  typedef PeriodicBinFinderInZ<float>   BinFinderType;
  BinFinderType theBinFinder;

  void init();

};

#endif
