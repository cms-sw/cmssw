#ifndef MeasurementTracker_H
#define MeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

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
class TrackingGeometry;
class SiStripRecHitMatcher;
class GluedGeomDet;

class MeasurementTracker : public MeasurementDetSystem {
public:

  //B.M. MeasurementTracker( const edm::EventSetup&, const edm::Event&);
  MeasurementTracker( const edm::EventSetup&);

  virtual ~MeasurementTracker() {}
 
  void update( const edm::Event&) const;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface
  virtual const MeasurementDet*       idToDet(const DetId& id) const;

private:

  typedef std::map<DetId,MeasurementDet*>   DetContainer;

  DetContainer                        theDetMap;
  std::vector<TkStripMeasurementDet*> theStripDets;
  std::vector<TkPixelMeasurementDet*> thePixelDets;
  std::vector<TkGluedMeasurementDet*> theGluedDets;
  const TrackingGeometry*             theTrackerGeom;
  const GeometricSearchTracker*       theGeometricSearchTracker;

  const StripClusterParameterEstimator* stripCPE;
  const PixelClusterParameterEstimator* pixelCPE;
  SiStripRecHitMatcher*           theHitMatcher;

  void initialize(const edm::EventSetup&);

  void addStripDet( const GeomDet* gd,
		    const StripClusterParameterEstimator* cpe);
  void addPixelDet( const GeomDet* gd,
		    const PixelClusterParameterEstimator* cpe);

  void addGluedDet( const GluedGeomDet* gd, SiStripRecHitMatcher* matcher);

  void addPixelDets( const TrackingGeometry::DetContainer& dets);

  void addStripDets( const TrackingGeometry::DetContainer& dets);

};

#endif
