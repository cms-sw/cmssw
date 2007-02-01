#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include <string>
#include <memory>

using namespace edm;

MeasurementTrackerESProducer::MeasurementTrackerESProducer(const edm::ParameterSet & p) 
{  
  pset_ = p;
  setWhatProduced(this);
}

MeasurementTrackerESProducer::~MeasurementTrackerESProducer() {}

boost::shared_ptr<MeasurementTracker> 
MeasurementTrackerESProducer::produce(const CkfComponentsRecord& iRecord)
{ 
  std::string pixelCPEName = pset_.getParameter<std::string>("PixelCPE");
  std::string stripCPEName = pset_.getParameter<std::string>("StripCPE");
  std::string matcherName  = pset_.getParameter<std::string>("HitMatcher");
  
  const SiStripNoises *ptr_stripNoises = 0;
  edm::ESHandle<SiStripNoises>	stripNoises;
  if (pset_.getParameter<bool>("UseStripNoiseDB")) {
     iRecord.getRecord<SiStripNoisesRcd>().get(stripNoises);
     ptr_stripNoises = stripNoises.product();	
  }

  const SiStripDetCabling *ptr_stripCabling = 0;
  edm::ESHandle<SiStripDetCabling>		stripCabling;
  if (pset_.getParameter<bool>("UseStripCablingDB")) {
    iRecord.getRecord<SiStripDetCablingRcd>().get(stripCabling);
    ptr_stripCabling = stripCabling.product();	
  }
  
  edm::ESHandle<PixelClusterParameterEstimator> pixelCPE;
  edm::ESHandle<StripClusterParameterEstimator> stripCPE;
  edm::ESHandle<SiStripRecHitMatcher>           hitMatcher;
  edm::ESHandle<TrackerGeometry>                trackerGeom;
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;

  
  iRecord.getRecord<TrackerCPERecord>().get(pixelCPEName,pixelCPE);
  iRecord.getRecord<TrackerCPERecord>().get(stripCPEName,stripCPE);
  iRecord.getRecord<TrackerCPERecord>().get(matcherName,hitMatcher);
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(trackerGeom);
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);
  
  
  _measurementTracker  = boost::shared_ptr<MeasurementTracker>(new MeasurementTracker(pset_,
										      pixelCPE.product(),
										      stripCPE.product(),
										      hitMatcher.product(),
										      trackerGeom.product(),
										      geometricSearchTracker.product(),
										      ptr_stripCabling,
										      ptr_stripNoises) ); 
  return _measurementTracker;
}


