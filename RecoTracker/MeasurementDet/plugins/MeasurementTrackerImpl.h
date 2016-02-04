#ifndef MeasurementTrackerImpl_H
#define MeasurementTrackerImpl_H

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

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


class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class TkGluedMeasurementDet;
class TkStackMeasurementDet;
class GeometricSearchTracker;
class SiStripRecHitMatcher;
class GluedGeomDet;
class StackGeomDet;
class SiPixelFedCabling;

class dso_hidden MeasurementTrackerImpl final : public MeasurementTracker {
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
                     int   pixelQualityDebugFlags);

  virtual ~MeasurementTrackerImpl();
 
  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

  const GeometricSearchTracker* geometricSearchTracker() const {return theGeometricSearchTracker;}

  /// MeasurementDetSystem interface  (won't be overloaded anymore)
  MeasurementDetWithData 
  idToDet(const DetId& id, const MeasurementTrackerEvent &data) const {
    return MeasurementDetWithData(*idToDetBare(id, data), data);
  }

  const MeasurementDet * 
  idToDetBare(const DetId& id, const MeasurementTrackerEvent &data) const {
    return findDet(id);
  }



  const MeasurementDet* 
  findDet(const DetId& id) const
  {
    auto it = theDetMap.find(id);
    if(it !=theDetMap.end()) {
      return it->second;
    }else{
      //throw exception;
    }
    
    return 0; //to avoid compile warning
  }

  typedef std::unordered_map<unsigned int,MeasurementDet*>   DetContainer;

  /// For debug only 
  const DetContainer& allDets() const {return theDetMap;}
  const std::vector<TkStripMeasurementDet>& stripDets() const {return theStripDets;}
  const std::vector<TkPixelMeasurementDet>& pixelDets() const {return thePixelDets;}
  const std::vector<TkGluedMeasurementDet>& gluedDets() const {return theGluedDets;}
  const std::vector<TkStackMeasurementDet>& stackDets() const {return theStackDets;}

  virtual const StMeasurementConditionSet & stripDetConditions() const { return theStDetConditions; }
  virtual const PxMeasurementConditionSet & pixelDetConditions() const { return thePxDetConditions; }

 protected:
  const edm::ParameterSet& pset_;
  const std::string name_;

  StMeasurementConditionSet theStDetConditions;
  PxMeasurementConditionSet thePxDetConditions;

  DetContainer                        theDetMap;

  std::vector<TkPixelMeasurementDet> thePixelDets;
  std::vector<TkStripMeasurementDet> theStripDets;
  std::vector<TkGluedMeasurementDet> theGluedDets;
  std::vector<TkStackMeasurementDet> theStackDets;

  const SiPixelFedCabling*              thePixelCabling;

  void initialize();
  void initStMeasurementConditionSet(std::vector<TkStripMeasurementDet> & stripDets);
  void initPxMeasurementConditionSet(std::vector<TkPixelMeasurementDet> & pixelDets);

  void addStripDet( const GeomDet* gd);
  void addPixelDet( const GeomDet* gd);

  void addGluedDet( const GluedGeomDet* gd);
  void addStackDet( const StackGeomDet* gd);

  void initGluedDet( TkGluedMeasurementDet & det);
  void initStackDet( TkStackMeasurementDet & det);

  void addDets( const TrackingGeometry::DetContainer& dets, bool subIsPixel);

  bool checkDets();


  void initializeStripStatus (const SiStripQuality *stripQuality, int qualityFlags, int qualityDebugFlags);

  void initializePixelStatus (const SiPixelQuality *stripQuality, const SiPixelFedCabling *pixelCabling, int qualityFlags, int qualityDebugFlags);
};

#endif
