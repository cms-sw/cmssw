#ifndef MeasurementTracker_H
#define MeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

// backward compatibility
#include "FWCore/Framework/interface/ESHandle.h"

class SiStripRecHitMatcher;


class MeasurementTracker : public MeasurementDetSystem {
public:
   enum QualityFlags { BadModules=1, // for everybody
                       /* Strips: */ BadAPVFibers=2, BadStrips=4, MaskBad128StripBlocks=8, 
                       /* Pixels: */ BadROCs=2 }; 

  MeasurementTracker(TrackerGeometry const *  trackerGeom,
		     GeometricSearchTracker const * geometricSearchTracker) : 
    theTrackerGeom(trackerGeom), theGeometricSearchTracker(geometricSearchTracker) {}



  virtual ~MeasurementTracker();
 
  virtual void update( const edm::Event&) const =0;
  virtual void updatePixels( const edm::Event&) const =0;
  virtual void updateStrips( const edm::Event&) const =0;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface
  virtual const MeasurementDet*       idToDet(const DetId& id) const =0;


  virtual void setClusterToSkip(const edm::InputTag & cluster, const edm::Event& event) const=0;
  virtual void unsetClusterToSkip() const=0;


protected:
  const TrackerGeometry*                theTrackerGeom;
  const GeometricSearchTracker*         theGeometricSearchTracker;


};

#endif // MeasurementTracker_H
