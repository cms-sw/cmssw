#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerESProducer.h"

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
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "RecoTracker/MeasurementDet/interface/OnDemandMeasurementTracker.h"

#include <string>
#include <memory>

using namespace edm;

MeasurementTrackerESProducer::MeasurementTrackerESProducer(const edm::ParameterSet & p) 
{  
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

MeasurementTrackerESProducer::~MeasurementTrackerESProducer() {}

boost::shared_ptr<MeasurementTracker> 
MeasurementTrackerESProducer::produce(const CkfComponentsRecord& iRecord)
{ 
  std::string pixelCPEName = pset_.getParameter<std::string>("PixelCPE");
  std::string stripCPEName = pset_.getParameter<std::string>("StripCPE");
  std::string matcherName  = pset_.getParameter<std::string>("HitMatcher");
  bool regional            = pset_.getParameter<bool>("Regional");  

  bool onDemand = pset_.getParameter<bool>("OnDemand");

  /* vvvv LEGACY, WILL BE REMOVED vvvv */
  const SiStripNoises *ptr_stripNoises = 0;
  edm::ESHandle<SiStripNoises>	stripNoises;
  if (pset_.getParameter<bool>("UseStripNoiseDB")) {
     iRecord.getRecord<SiStripNoisesRcd>().get(stripNoises);
     ptr_stripNoises = stripNoises.product();	
  }
  /* ^^^^ LEGACY, WILL BE REMOVED ^^^^ */

  const SiStripQuality *ptr_stripQuality = 0;
  edm::ESHandle<SiStripQuality>	stripQuality;

  if (pset_.getParameter<bool>("UseStripModuleQualityDB")) {
    iRecord.getRecord<SiStripQualityRcd>().get(stripQuality);
    ptr_stripQuality = stripQuality.product();	
  }
  
  edm::ESHandle<PixelClusterParameterEstimator> pixelCPE;
  edm::ESHandle<StripClusterParameterEstimator> stripCPE;
  edm::ESHandle<SiStripRecHitMatcher>           hitMatcher;
  edm::ESHandle<TrackerGeometry>                trackerGeom;
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;

  
  iRecord.getRecord<TkPixelCPERecord>().get(pixelCPEName,pixelCPE);
  iRecord.getRecord<TkStripCPERecord>().get(stripCPEName,stripCPE);
  iRecord.getRecord<TkStripCPERecord>().get(matcherName,hitMatcher);
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(trackerGeom);
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);
  
  if (!onDemand){
  _measurementTracker  = boost::shared_ptr<MeasurementTracker>(new MeasurementTracker(pset_,
										      pixelCPE.product(),
										      stripCPE.product(),
										      hitMatcher.product(),
										      trackerGeom.product(),
										      geometricSearchTracker.product(),
										      ptr_stripQuality,
										      ptr_stripNoises,
										      regional) ); 
  }
  else{
    const SiStripRegionCabling * ptr_stripRegionCabling =0;
    //get regional cabling
    edm::ESHandle<SiStripRegionCabling> rcabling;
    iRecord.getRecord<SiStripRegionCablingRcd>().get(rcabling);
    ptr_stripRegionCabling = rcabling.product();

    _measurementTracker  = boost::shared_ptr<MeasurementTracker>( new OnDemandMeasurementTracker(pset_,
												 pixelCPE.product(),
												 stripCPE.product(),
												 hitMatcher.product(),
												 trackerGeom.product(),
												 geometricSearchTracker.product(),
												 ptr_stripQuality,
												 ptr_stripNoises,
												 ptr_stripRegionCabling,
												 regional) );
    
  }
  return _measurementTracker;
}


