#ifndef DetLayers_MuRingForwardLayer_H
#define DetLayers_MuRingForwardLayer_H

/** \class MuRingForwardLayer
 *  A plane composed of disks (MuRingForwardDisk). Represents forward muon CSC/RPC stations.
 *
 *  $Date: 2006/05/16 09:43:00 $
 *  $Revision: 1.3 $
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

  virtual const vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
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


  virtual bool hasGroups() const;


  // DetLayer interface

  virtual Module module() const;


  // Extension of the interface

  /// Return the vector of rings.
  virtual const vector<const ForwardDetRing*>& rings() const {return theRings;}


 private:  
  vector<const ForwardDetRing*> theRings;
  vector <const GeometricSearchDet*> theComponents; // duplication of the above
  vector<const GeomDet*> theBasicComps; // All chambers
  BaseBinFinder<double> * theBinFinder;
  bool isOverlapping;

};
#endif

