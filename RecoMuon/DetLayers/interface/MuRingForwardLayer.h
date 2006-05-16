#ifndef DetLayers_MuRingForwardLayer_H
#define DetLayers_MuRingForwardLayer_H

/** \class MuRingForwardLayer
 *  A plane composed of disks (MuRingForwardDisk). Represents forward muon CSC/RPC stations.
 *
 *  $Date: 2006/04/25 17:03:23 $
 *  $Revision: 1.2 $
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

  MuRingForwardLayer(vector<const ForwardDetRing*>& rings);

  virtual ~MuRingForwardLayer();


  // GeometricSearchDet interface

  virtual vector<const GeomDet*> basicComponents() const {return theBasicComps;}
  
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


  virtual bool hasGroups() const;


  // DetLayer interface

  virtual Module module() const;


  // Extension of the interface

  /// Return the vector of rings.
  virtual const vector<const ForwardDetRing*>& rings() const {return theRings;}


 private:  
  vector<const ForwardDetRing*> theRings;
  vector<const GeomDet*> theBasicComps;
  BaseBinFinder<double> * theBinFinder;
  bool isOverlapping;

};
#endif

