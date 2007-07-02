#ifndef MeasurementTracker_H
#define MeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include <map>
#include <vector>

class StrictWeakOrdering{
 public:
  bool operator() ( uint32_t p,const uint32_t& i) const {return p < i;}
};
class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class TkGluedMeasurementDet;
class GeometricSearchTracker;
class SiStripRecHitMatcher;
class GluedGeomDet;

class MeasurementTracker : public MeasurementDetSystem {
public:

  MeasurementTracker(const edm::ParameterSet&              conf,
		     const PixelClusterParameterEstimator* pixelCPE,
		     const StripClusterParameterEstimator* stripCPE,
		     const SiStripRecHitMatcher*  hitMatcher,
		     const TrackerGeometry*  trackerGeom,
		     const GeometricSearchTracker* geometricSearchTracker,
		     const SiStripDetCabling *stripCabling,
		     const SiStripNoises *stripNoises,
                     bool  isRegional=false);

  virtual ~MeasurementTracker() { if (dummyStripNoises) delete dummyStripNoises; }
 
  void update( const edm::Event&) const;
  void updatePixels( const edm::Event&) const;
  void updateStrips( const edm::Event&) const;

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
  const edm::ParameterSet& pset_;

  mutable unsigned int lastEventNumberPixels;
  mutable unsigned int lastEventNumberStrips;
  mutable unsigned int lastRunNumberPixels;
  mutable unsigned int lastRunNumberStrips;


  mutable DetContainer                        theDetMap;
  mutable std::vector<TkStripMeasurementDet*> theStripDets;
  mutable std::vector<TkPixelMeasurementDet*> thePixelDets;
  mutable std::vector<TkGluedMeasurementDet*> theGluedDets;

  const PixelClusterParameterEstimator* thePixelCPE;
  const StripClusterParameterEstimator* theStripCPE;
  const SiStripRecHitMatcher*           theHitMatcher;
  const TrackerGeometry*                theTrackerGeom;
  const GeometricSearchTracker*         theGeometricSearchTracker;
  mutable SiStripNoises*                dummyStripNoises;  // not const

  bool isRegional_;

  void initialize() const;

  void addStripDet( const GeomDet* gd,
		    const StripClusterParameterEstimator* cpe) const;
  void addPixelDet( const GeomDet* gd,
		    const PixelClusterParameterEstimator* cpe) const;

  void addGluedDet( const GluedGeomDet* gd, const SiStripRecHitMatcher* matcher) const;

  void addPixelDets( const TrackingGeometry::DetContainer& dets) const;

  void addStripDets( const TrackingGeometry::DetContainer& dets) const;

  void initializeStripStatus (const SiStripDetCabling *stripCabling) const;
  void initializeStripNoises (const SiStripNoises *stripNoises) const;
};

#endif
