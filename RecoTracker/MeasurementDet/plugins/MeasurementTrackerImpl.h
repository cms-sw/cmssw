#ifndef MeasurementTrackerImpl_H
#define MeasurementTrackerImpl_H

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TkMeasurementDetSet.h"

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
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include <map>
#include <unordered_map>
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
class SiPixelFedCabling;

class MeasurementTrackerImpl : public MeasurementTracker {
public:
   enum QualityFlags { BadModules=1, // for everybody
                       /* Strips: */ BadAPVFibers=2, BadStrips=4, MaskBad128StripBlocks=8, 
                       /* Pixels: */ BadROCs=2 }; 

  MeasurementTrackerImpl(const edm::ParameterSet&              conf,
		     const PixelClusterParameterEstimator* pixelCPE,
		     const StripClusterParameterEstimator* stripCPE,
		     const SiStripRecHitMatcher*  hitMatcher,
		     const TrackerGeometry*  trackerGeom,
		     const GeometricSearchTracker* geometricSearchTracker,
                     const SiStripQuality *stripQuality,
                     int   stripQualityFlags,
                     int   stripQualityDebugFlags,
                     const SiPixelQuality *pixelQuality,
                     const SiPixelFedCabling *pixelCabling,
                     int   pixelQualityFlags,
                     int   pixelQualityDebugFlags,
		     bool  isRegional=false);

  virtual ~MeasurementTrackerImpl();
 
  virtual  void update( const edm::Event&) const;
  void updatePixels( const edm::Event&) const;
  void updateStrips( const edm::Event&) const;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface
  virtual const MeasurementDet*       idToDet(const DetId& id) const;

  TkStripMeasurementDet * concreteDetUpdatable(DetId id) const;

  typedef std::unordered_map<unsigned int,MeasurementDet*>   DetContainer;

  /// For debug only 
  const DetContainer& allDets() const {return theDetMap;}
  const std::vector<TkStripMeasurementDet*>& stripDets() const {return theStripDets;}
  const std::vector<TkPixelMeasurementDet*>& pixelDets() const {return thePixelDets;}
  const std::vector<TkGluedMeasurementDet*>& gluedDets() const {return theGluedDets;}

  void setClusterToSkip(const edm::InputTag & cluster, const edm::Event& event) const;
  void unsetClusterToSkip() const;
  
 protected:
  const edm::ParameterSet& pset_;
  const std::string name_;

  TkMeasurementDetSet theDets;

  mutable DetContainer                        theDetMap;

  mutable std::vector<TkPixelMeasurementDet*> thePixelDets;
  mutable std::vector<TkGluedMeasurementDet*> theGluedDets;
  
  mutable std::vector<bool> thePixelsToSkip;
  mutable std::vector<bool> theStripsToSkip;

  const PixelClusterParameterEstimator* thePixelCPE;
  const SiPixelFedCabling*              thePixelCabling;

  const std::vector<edm::InputTag>      theInactivePixelDetectorLabels;
  const std::vector<edm::InputTag>      theInactiveStripDetectorLabels;

  bool selfUpdateSkipClusters_;

  void initialize() const;

  void addStripDet( const GeomDet* gd,
		    const StripClusterParameterEstimator* cpe) const;
  void addPixelDet( const GeomDet* gd,
		    const PixelClusterParameterEstimator* cpe) const;

  void addGluedDet( const GluedGeomDet* gd, const SiStripRecHitMatcher* matcher) const;

  void addPixelDets( const TrackingGeometry::DetContainer& dets) const;

  void addStripDets( const TrackingGeometry::DetContainer& dets) const;

  void initializeStripStatus (const SiStripQuality *stripQuality, int qualityFlags, int qualityDebugFlags) const;

  void initializePixelStatus (const SiPixelQuality *stripQuality, const SiPixelFedCabling *pixelCabling, int qualityFlags, int qualityDebugFlags) const;

  void getInactiveStrips(const edm::Event& event,std::vector<uint32_t> & rawInactiveDetIds) const;
};

#endif
