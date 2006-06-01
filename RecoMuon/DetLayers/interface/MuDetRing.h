#ifndef DetLayers_MuDetRing_H
#define DetLayers_MuDetRing_H

/** \class MuDetRing
 *  A ring of periodic, possibly overlapping vertical detectors.
 *  Designed for forward muon CSC/RPC chambers.
 *
 *  $Date: 2006/04/12 13:23:53 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - INFN Torino
 */

#include "TrackingTools/DetLayers/interface/ForwardDetRingOneZ.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

class GeomDet;

class MuDetRing : public ForwardDetRingOneZ {
 public:

  /// Construct from iterators on GeomDet*
  MuDetRing(vector<const GeomDet*>::const_iterator first,
	    vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of GeomDet*
  MuDetRing(const vector<const GeomDet*>& dets);

  virtual ~MuDetRing();


  // GeometricSearchDet interface

  virtual const vector<const GeometricSearchDet*>& components() const;

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


  // FIXME: should be implemented (overlaps in forward CSC and RPC)
  virtual bool hasGroups() const {return false;}

 private:
  typedef PeriodicBinFinderInPhi<float>   BinFinderType;
  BinFinderType theBinFinder;

  void init();

};
#endif

