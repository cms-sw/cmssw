#ifndef DetLayers_MuRodBarrelLayer_H
#define DetLayers_MuRodBarrelLayer_H

/** \class MuRodBarrelLayer
 *  A cylinder composed of rods. Represents barrel muon DT/RPC stations.
 *
 *  $Date: 2006/07/25 17:10:27 $
 *  $Revision: 1.8 $
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

  virtual ~MuRodBarrelLayer();

  // GeometricSearchDet interface

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}

  virtual const std::vector<const GeometricSearchDet*>& components() const;  

  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;
  
  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const;


  // DetLayer interface
  virtual SubDetector subDetector() const;

  // Extension of the interface

  /// Return the vector of rods.
  virtual const std::vector<const DetRod*>& rods() const {return theRods;}


private:

  float xError(const TrajectoryStateOnSurface& tsos,
	       const MeasurementEstimator& est) const;

  std::vector<const DetRod*> theRods;
  std::vector <const GeometricSearchDet*> theComponents; // duplication of the above
  std::vector<const GeomDet*> theBasicComps; // All chambers
  BaseBinFinder<double> * theBinFinder;
  bool isOverlapping;
};

#endif
