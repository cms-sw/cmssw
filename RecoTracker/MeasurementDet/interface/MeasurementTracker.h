#ifndef MeasurementTracker_H
#define MeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <map>
#include <vector>

class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class TkGluedMeasurementDet;
class GeometricSearchTracker;
class SiStripRecHitMatcher;
class GluedGeomDet;

class MeasurementTracker : public MeasurementDetSystem {
public:

  MeasurementTracker(const PixelClusterParameterEstimator* pixelCPE,
		     const StripClusterParameterEstimator* stripCPE,
		     const SiStripRecHitMatcher*  hitMatcher,
		     const TrackerGeometry*  trackerGeom,
		     const GeometricSearchTracker* geometricSearchTracker);

  virtual ~MeasurementTracker() {}
 
  void update( const edm::Event&) const;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface
  virtual const MeasurementDet*       idToDet(const DetId& id) const;

  typedef std::map<DetId,MeasurementDet*>   DetContainer;

  /// For debug only 
  const DetContainer& allDets() const {return theDetMap;}
  const std::vector<TkStripMeasurementDet*>& stripDets() const {return theStripDets;}
  const std::vector<TkPixelMeasurementDet*>& pixelDets() const {return thePixelDets;}
  const std::vector<TkGluedMeasurementDet*>& gluedDets() const {return theGluedDets;}


private:
  mutable unsigned int lastEventNumber;
  mutable unsigned int lastRunNumber;

  mutable DetContainer                        theDetMap;
  mutable std::vector<TkStripMeasurementDet*> theStripDets;
  mutable std::vector<TkPixelMeasurementDet*> thePixelDets;
  mutable std::vector<TkGluedMeasurementDet*> theGluedDets;

  const PixelClusterParameterEstimator* thePixelCPE;
  const StripClusterParameterEstimator* theStripCPE;
  const SiStripRecHitMatcher*           theHitMatcher;
  const TrackerGeometry*                theTrackerGeom;
  const GeometricSearchTracker*         theGeometricSearchTracker;

  void initialize() const;

  void addStripDet( const GeomDet* gd,
		    const StripClusterParameterEstimator* cpe) const;
  void addPixelDet( const GeomDet* gd,
		    const PixelClusterParameterEstimator* cpe) const;

  void addGluedDet( const GluedGeomDet* gd, const SiStripRecHitMatcher* matcher) const;

  void addPixelDets( const TrackingGeometry::DetContainer& dets) const;

  void addStripDets( const TrackingGeometry::DetContainer& dets) const;

};

#endif
