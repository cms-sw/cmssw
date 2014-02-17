#ifndef DetLayers_MuRingForwardDoubleLayer_H
#define DetLayers_MuRingForwardDoubleLayer_H

/** \class MuRingForwardDoubleLayer
 *  A plane composed two layers of disks. Represents forward muon CSC stations.
 *
 *  $Date: 2007/12/19 09:48:34 $
 *  $Revision: 1.2 $
 *  \author R. Wilkinson
 *
 */

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "RecoMuon/DetLayers/interface/MuRingForwardLayer.h"

class ForwardDetRing;
class ForwardDetRingBuilder;
class GeomDet;

class MuRingForwardDoubleLayer : public RingedForwardLayer {

 public:  

  /// Constructor, takes ownership of pointers
  MuRingForwardDoubleLayer(const std::vector<const ForwardDetRing*>& frontRings,  
                           const std::vector<const ForwardDetRing*>& backRings);

  virtual ~MuRingForwardDoubleLayer() {}


  // GeometricSearchDet interface

  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComponents;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const {return theComponents;}

  bool isInsideOut(const TrajectoryStateOnSurface&tsos) const;

  // tries closest layer first
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&,
              const MeasurementEstimator&) const;

  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const;
  
  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const;


  virtual bool hasGroups() const {return true;}


  // DetLayer interface
  virtual SubDetector subDetector() const {return theBackLayer.subDetector();}


  // Extension of the interface

  /// Return the vector of rings.
  virtual const std::vector<const ForwardDetRing*>& rings() const {return theRings;}

  bool isCrack(const GlobalPoint & gp) const;

  const MuRingForwardLayer * frontLayer() const {return &theFrontLayer;}
  const MuRingForwardLayer * backLayer() const {return &theBackLayer;}

  void selfTest() const;
 protected:
    virtual BoundDisk * computeSurface();
 private:  
  MuRingForwardLayer theFrontLayer;
  MuRingForwardLayer theBackLayer;
  std::vector<const ForwardDetRing*> theRings;
  std::vector <const GeometricSearchDet*> theComponents; // duplication of the above
  std::vector<const GeomDet*> theBasicComponents; // All chambers

};
#endif

