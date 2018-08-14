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
class StMeasurementConditionSet;
class PxMeasurementConditionSet;
class Phase2OTMeasurementConditionSet;

class MeasurementTracker : public MeasurementDetSystem {
public:
   enum QualityFlags { BadModules=1, // for everybody
                       /* Strips: */ BadAPVFibers=2, BadStrips=4, MaskBad128StripBlocks=8, 
                       /* Pixels: */ BadROCs=2 }; 

  MeasurementTracker(TrackerGeometry const *  trackerGeom,
		     GeometricSearchTracker const * geometricSearchTracker) : 
    theTrackerGeom(trackerGeom), theGeometricSearchTracker(geometricSearchTracker) {}



  ~MeasurementTracker() override;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface
  MeasurementDetWithData idToDet(const DetId& id, const MeasurementTrackerEvent &data) const override = 0;

  /// Provide templates to be filled in
  virtual const StMeasurementConditionSet & stripDetConditions() const = 0;
  virtual const PxMeasurementConditionSet & pixelDetConditions() const = 0;
  virtual const Phase2OTMeasurementConditionSet & phase2DetConditions() const = 0;

protected:
  const TrackerGeometry*                theTrackerGeom;
  const GeometricSearchTracker*         theGeometricSearchTracker;


};

#endif // MeasurementTracker_H
