#include "RecoTracker/MeasurementDet/plugins/MeasurementTrackerESProducer.h"

#include "MeasurementTrackerImpl.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
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
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include <string>
#include <memory>

using namespace edm;

MeasurementTrackerESProducer::MeasurementTrackerESProducer(const edm::ParameterSet & p) 
{  
  std::string myname = p.getParameter<std::string>("ComponentName");
  pixelCPEName = p.getParameter<std::string>("PixelCPE");
  stripCPEName = p.getParameter<std::string>("StripCPE");
  matcherName  = p.getParameter<std::string>("HitMatcher");

  //FIXME:: just temporary solution for phase2!
  phase2TrackerCPEName = "";
  if (p.existsAs<std::string>("Phase2StripCPE")) {
    phase2TrackerCPEName = p.getParameter<std::string>("Phase2StripCPE");
  }

  pset_ = p;
  setWhatProduced(this,myname);
}

MeasurementTrackerESProducer::~MeasurementTrackerESProducer() {}

std::shared_ptr<MeasurementTracker> 
MeasurementTrackerESProducer::produce(const CkfComponentsRecord& iRecord)
{ 

  // ========= SiPixelQuality related tasks =============
  const SiPixelQuality    *ptr_pixelQuality = nullptr;
  const SiPixelFedCabling *ptr_pixelCabling = nullptr;
  int   pixelQualityFlags = 0;
  int   pixelQualityDebugFlags = 0;
  edm::ESHandle<SiPixelQuality>	      pixelQuality;
  edm::ESHandle<SiPixelFedCablingMap> pixelCabling;

  if (pset_.getParameter<bool>("UsePixelModuleQualityDB")) {
    pixelQualityFlags += MeasurementTracker::BadModules;
    if (pset_.getUntrackedParameter<bool>("DebugPixelModuleQualityDB", false)) {
        pixelQualityDebugFlags += MeasurementTracker::BadModules;
    }
  }
  if (pset_.getParameter<bool>("UsePixelROCQualityDB")) {
    pixelQualityFlags += MeasurementTracker::BadROCs;
    if (pset_.getUntrackedParameter<bool>("DebugPixelROCQualityDB", false)) {
        pixelQualityDebugFlags += MeasurementTracker::BadROCs;
    }
  }


  if (pixelQualityFlags != 0) {
    iRecord.getRecord<SiPixelQualityRcd>().get(pixelQuality);
    ptr_pixelQuality = pixelQuality.product();
    iRecord.getRecord<SiPixelFedCablingMapRcd>().get(pixelCabling);
    ptr_pixelCabling = pixelCabling.product();
  }
  
  // ========= SiStripQuality related tasks =============
  const SiStripQuality *ptr_stripQuality = nullptr;
  int   stripQualityFlags = 0;
  int   stripQualityDebugFlags = 0;
  edm::ESHandle<SiStripQuality>	stripQuality;

  if (pset_.getParameter<bool>("UseStripModuleQualityDB")) {
    stripQualityFlags += MeasurementTracker::BadModules;
    if (pset_.getUntrackedParameter<bool>("DebugStripModuleQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadModules;
    }
  }
  if (pset_.getParameter<bool>("UseStripAPVFiberQualityDB")) {
    stripQualityFlags += MeasurementTracker::BadAPVFibers;
    if (pset_.getUntrackedParameter<bool>("DebugStripAPVFiberQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadAPVFibers;
    }
    if (pset_.getParameter<bool>("MaskBadAPVFibers")) {
        stripQualityFlags += MeasurementTracker::MaskBad128StripBlocks;
    }
  }
  if (pset_.getParameter<bool>("UseStripStripQualityDB")) {
    stripQualityFlags += MeasurementTracker::BadStrips;
    if (pset_.getUntrackedParameter<bool>("DebugStripStripQualityDB", false)) {
        stripQualityDebugFlags += MeasurementTracker::BadStrips;
    }
  }

  if (stripQualityFlags != 0) {
    std::string siStripQualityLabel = pset_.getParameter<std::string>("SiStripQualityLabel");
    iRecord.getRecord<SiStripQualityRcd>().get(siStripQualityLabel, stripQuality);
    ptr_stripQuality = stripQuality.product();
  }
  
  edm::ESHandle<PixelClusterParameterEstimator> pixelCPE;
  edm::ESHandle<StripClusterParameterEstimator> stripCPE;
  edm::ESHandle<SiStripRecHitMatcher>           hitMatcher;
  edm::ESHandle<TrackerTopology>                trackerTopology;
  edm::ESHandle<TrackerGeometry>                trackerGeom;
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > phase2TrackerCPE;
  
  iRecord.getRecord<TkPixelCPERecord>().get(pixelCPEName,pixelCPE);
  iRecord.getRecord<TkStripCPERecord>().get(stripCPEName,stripCPE);
  iRecord.getRecord<TkStripCPERecord>().get(matcherName,hitMatcher);
  iRecord.getRecord<TrackerTopologyRcd>().get(trackerTopology);
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(trackerGeom);
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);

  if(phase2TrackerCPEName != ""){
      iRecord.getRecord<TkStripCPERecord>().get(phase2TrackerCPEName,phase2TrackerCPE);
      _measurementTracker  = std::make_shared<MeasurementTrackerImpl>(pset_,
							          pixelCPE.product(),
							          stripCPE.product(),
							          hitMatcher.product(),
							          trackerTopology.product(),
							          trackerGeom.product(),
							          geometricSearchTracker.product(),
							          ptr_stripQuality,
                                                                  stripQualityFlags,
                                                                  stripQualityDebugFlags,
							          ptr_pixelQuality,
							          ptr_pixelCabling,
                                                                  pixelQualityFlags,
                                                                  pixelQualityDebugFlags,
							          phase2TrackerCPE.product());
  } else {
      _measurementTracker  = std::make_shared<MeasurementTrackerImpl>(pset_,
							          pixelCPE.product(),
							          stripCPE.product(),
							          hitMatcher.product(),
							          trackerTopology.product(),
							          trackerGeom.product(),
							          geometricSearchTracker.product(),
							          ptr_stripQuality,
                                                                  stripQualityFlags,
                                                                  stripQualityDebugFlags,
							          ptr_pixelQuality,
                                                                  ptr_pixelCabling,
                                                                  pixelQualityFlags,
                                                                  pixelQualityDebugFlags);
  }
  return _measurementTracker;
}


