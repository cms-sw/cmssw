#ifndef DetLayers_MuRodBarrelLayer_H
#define DetLayers_MuRodBarrelLayer_H

/** \class MuRodBarrelLayer
 *  A cylinder composed of rods. Represents barrel muon DT/RPC stations.
 *
 *  $Date: 2006/05/16 09:43:00 $
 *  $Revision: 1.3 $
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

  MuRodBarrelLayer(vector<const DetRod*>& rods);

  virtual ~MuRodBarrelLayer();

  // GeometricSearchDet interface

  virtual const vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const vector<const GeometricSearchDet*>& components() const;
  
  virtual pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface& ts, const Propagator& prop, 
	      const MeasurementEstimator&) const;

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

  /// Return the vector of rods.
  virtual const vector<const DetRod*>& rods() const {return theRods;}


private:
  vector<const DetRod*> theRods;
  vector <const GeometricSearchDet*> theComponents; // duplication of the above
  vector<const GeomDet*> theBasicComps; // All chambers
  BaseBinFinder<double> * theBinFinder;
  bool isOverlapping;
};

#endif
