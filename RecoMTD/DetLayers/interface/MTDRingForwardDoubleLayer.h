#ifndef DetLayers_MTDRingForwardDoubleLayer_H
#define DetLayers_MTDRingForwardDoubleLayer_H

/** \class MTDRingForwardDoubleLayer
 *  A plane composed two layers of disks. The Endcap Timing Layer.
 *
 *  \author L. Gray
 *
 */

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Utilities/BinningTools/interface/BaseBinFinder.h"
#include "RecoMTD/DetLayers/interface/MTDRingForwardLayer.h"

class ForwardDetRing;
class ForwardDetRingBuilder;
class GeomDet;

class MTDRingForwardDoubleLayer : public RingedForwardLayer {

 public:  

  /// Constructor, takes ownership of pointers
  MTDRingForwardDoubleLayer(const std::vector<const ForwardDetRing*>& frontRings,  
                           const std::vector<const ForwardDetRing*>& backRings);

  ~MTDRingForwardDoubleLayer() override {}


  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override {return theBasicComponents;}
  
  const std::vector<const GeometricSearchDet*>& components() const override {return theComponents;}

  bool isInsideOut(const TrajectoryStateOnSurface&tsos) const;

  // tries closest layer first
  std::pair<bool, TrajectoryStateOnSurface>
  compatible( const TrajectoryStateOnSurface&, const Propagator&,
              const MeasurementEstimator&) const override;

  std::vector<DetWithState> 
  compatibleDets( const TrajectoryStateOnSurface& startingState,
		  const Propagator& prop, 
		  const MeasurementEstimator& est) const override;
  
  std::vector<DetGroup> 
  groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
			 const Propagator& prop,
			 const MeasurementEstimator& est) const override;


  // DetLayer interface
  SubDetector subDetector() const override {return theBackLayer.subDetector();}


  // Extension of the interface

  /// Return the vector of rings.
  virtual const std::vector<const ForwardDetRing*>& rings() const {return theRings;}

  bool isCrack(const GlobalPoint & gp) const;

  const MTDRingForwardLayer * frontLayer() const {return &theFrontLayer;}
  const MTDRingForwardLayer * backLayer() const {return &theBackLayer;}

  void selfTest() const;
 protected:
    BoundDisk * computeSurface() override;
 private:  
  MTDRingForwardLayer theFrontLayer;
  MTDRingForwardLayer theBackLayer;
  std::vector<const ForwardDetRing*> theRings;
  std::vector <const GeometricSearchDet*> theComponents; // duplication of the above
  std::vector<const GeomDet*> theBasicComponents; // All chambers

};
#endif

