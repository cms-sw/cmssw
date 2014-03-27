#ifndef DetLayers_MuRingForwardDoubleLayer_H
#define DetLayers_MuRingForwardDoubleLayer_H

/** \class MuRingForwardDoubleLayer
 *  A plane composed two layers of disks. Represents forward muon CSC stations.
 *
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

  virtual const std::vector<const GeomDet*>& basicComponents() const override {return theBasicComponents;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const override {return theComponents;}

  bool isInsideOut(const TrajectoryStateOnSurface&tsos) const;

  // tries closest layer first
  virtual std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, Propagator&,
              const MeasurementEstimator&) const override;

  virtual std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  Propagator& prop, 
		  const MeasurementEstimator& est) const override;
  
  virtual std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 Propagator& prop,
			 const MeasurementEstimator& est) const override;


  // DetLayer interface
  virtual SubDetector subDetector() const override {return theBackLayer.subDetector();}


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

