#include "MeasurementTrackerImpl.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"  

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TkStripMeasurementDet.h"
#include "TkPixelMeasurementDet.h"
#include "TkGluedMeasurementDet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include <iostream>
#include <typeinfo>
#include <map>
#include <algorithm>


//

using namespace std;

namespace {

  struct CmpTKD {
    bool operator()(MeasurementDet const* rh, MeasurementDet const * lh) {
      return rh->fastGeomDet().geographicalId().rawId() < lh->fastGeomDet().geographicalId().rawId();
    }
    bool operator()(MeasurementDet const & rh, MeasurementDet const &  lh) {
      return rh.fastGeomDet().geographicalId().rawId() < lh.fastGeomDet().geographicalId().rawId();
    }
  };

  template<typename TKD>
  void sortTKD( std::vector<TKD*> & det) {
    std::sort(det.begin(),det.end(),CmpTKD());
  }
  template<typename TKD>
  void sortTKD( std::vector<TKD> & det) {
    std::sort(det.begin(),det.end(),CmpTKD());
  }

}


MeasurementTrackerImpl::MeasurementTrackerImpl(const edm::ParameterSet&              conf,
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
                                       int   pixelQualityDebugFlags) :
  MeasurementTracker(trackerGeom,geometricSearchTracker),
  pset_(conf),
  name_(conf.getParameter<std::string>("ComponentName")),
  theStDetConditions(hitMatcher,stripCPE),
  thePxDetConditions(pixelCPE)
{
  this->initialize();
  this->initializeStripStatus(stripQuality, stripQualityFlags, stripQualityDebugFlags);
  this->initializePixelStatus(pixelQuality, pixelCabling, pixelQualityFlags, pixelQualityDebugFlags);
}

MeasurementTrackerImpl::~MeasurementTrackerImpl()
{
}


void MeasurementTrackerImpl::initialize()
{  
  addPixelDets( theTrackerGeom->detsPXB());
  addPixelDets( theTrackerGeom->detsPXF());

  addStripDets( theTrackerGeom->detsTIB());
  addStripDets( theTrackerGeom->detsTID());
  addStripDets( theTrackerGeom->detsTOB());
  addStripDets( theTrackerGeom->detsTEC());  

  // fist all stripdets
  sortTKD(theStripDets);
  initStMeasurementConditionSet(theStripDets);
  for (unsigned int i=0; i!=theStripDets.size(); ++i)
    theDetMap[theStDetConditions.id(i)] = &theStripDets[i];
  
  // now the glued dets
  sortTKD(theGluedDets);
  for (unsigned int i=0; i!=theGluedDets.size(); ++i)
    initGluedDet(theGluedDets[i]);

  // and then the pixels, at last
  sortTKD(thePixelDets);
  initPxMeasurementConditionSet(thePixelDets);
  for (unsigned int i=0; i!=thePixelDets.size(); ++i)
    theDetMap[thePxDetConditions.id(i)] = &thePixelDets[i];

}

void MeasurementTrackerImpl::initStMeasurementConditionSet(std::vector<TkStripMeasurementDet> & stripDets)
{
  // assume vector is full and ordered!
  int size = stripDets.size();
  theStDetConditions.init(size);
  for (int i=0; i!=size; ++i) {
    auto & mdet =  stripDets[i]; 
    mdet.setIndex(i);
    //intialize the detId !
    theStDetConditions.id_[i] = mdet.specificGeomDet().geographicalId().rawId();
    theStDetConditions.subId_[i]=SiStripDetId(theStDetConditions.id_[i]).subdetId()-3;
    //initalize the total number of strips
    theStDetConditions.totalStrips_[i] =  mdet.specificGeomDet().specificTopology().nstrips();
  }
}

void MeasurementTrackerImpl::initPxMeasurementConditionSet(std::vector<TkPixelMeasurementDet> & pixelDets)
{
  // assume vector is full and ordered!
  int size = pixelDets.size();
  thePxDetConditions.init(size);

  for (int i=0; i!=size; ++i) {
    auto & mdet =  pixelDets[i]; 
    mdet.setIndex(i);
    thePxDetConditions.id_[i] = mdet.specificGeomDet().geographicalId().rawId();
  }
}




void MeasurementTrackerImpl::addPixelDets( const TrackingGeometry::DetContainer& dets)
{
  for (TrackerGeometry::DetContainer::const_iterator gd=dets.begin();
       gd != dets.end(); gd++) {
    addPixelDet(*gd);
  }  
}

void MeasurementTrackerImpl::addStripDets( const TrackingGeometry::DetContainer& dets)
{
  for (TrackerGeometry::DetContainer::const_iterator gd=dets.begin();
       gd != dets.end(); gd++) {

    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(*gd);

    //    StripSubdetector stripId( (**gd).geographicalId());
    //     bool isDetUnit( gdu != 0);
    //     cout << "StripSubdetector glued? " << stripId.glued() 
    // 	 << " is DetUnit? " << isDetUnit << endl;

    if (gdu != 0) {
      addStripDet(*gd);
    }
    else {
      const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*>(*gd);
      if (gluedDet == 0) {
	throw MeasurementDetException("MeasurementTracker ERROR: GeomDet neither DetUnit nor GluedDet");
      }
      addGluedDet(gluedDet);
    }  
  }
}

void MeasurementTrackerImpl::addStripDet( const GeomDet* gd)
{
  try {
    theStripDets.push_back(TkStripMeasurementDet( gd, theStDetConditions ));
  }
  catch(MeasurementDetException& err){
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what() ;
  }
}

void MeasurementTrackerImpl::addPixelDet( const GeomDet* gd)
{
  try {
    thePixelDets.push_back(TkPixelMeasurementDet( gd, thePxDetConditions ));
  }
  catch(MeasurementDetException& err){
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what() ;
  }
}

void MeasurementTrackerImpl::addGluedDet( const GluedGeomDet* gd)
{
  theGluedDets.push_back(TkGluedMeasurementDet( gd, theStDetConditions.matcher(), theStDetConditions.stripCPE() ));
}

void MeasurementTrackerImpl::initGluedDet( TkGluedMeasurementDet & det)
{
  const GluedGeomDet& gd = det.specificGeomDet();
  const MeasurementDet* monoDet = findDet( gd.monoDet()->geographicalId());
  const MeasurementDet* stereoDet = findDet( gd.stereoDet()->geographicalId());
  if (monoDet == 0 || stereoDet == 0) {
    edm::LogError("MeasurementDet") << "MeasurementTracker ERROR: GluedDet components not found as MeasurementDets ";
    throw MeasurementDetException("MeasurementTracker ERROR: GluedDet components not found as MeasurementDets");
  }
  det.init(monoDet,stereoDet);
  theDetMap[gd.geographicalId()] = &det;
}

void MeasurementTrackerImpl::initializeStripStatus(const SiStripQuality *quality, int qualityFlags, int qualityDebugFlags) {
  edm::ParameterSet cutPset = pset_.getParameter<edm::ParameterSet>("badStripCuts");
  if (qualityFlags & BadStrips) {
     typedef StMeasurementConditionSet::BadStripCuts BadStripCuts;
     theStDetConditions.badStripCuts_[SiStripDetId::TIB-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TIB"));
     theStDetConditions.badStripCuts_[SiStripDetId::TOB-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TOB"));
     theStDetConditions.badStripCuts_[SiStripDetId::TID-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TID"));
     theStDetConditions.badStripCuts_[SiStripDetId::TEC-3] = BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TEC"));
  }
  theStDetConditions.setMaskBad128StripBlocks((qualityFlags & MaskBad128StripBlocks) != 0);
  
  
  if ((quality != 0) && (qualityFlags != 0))  {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0; 
    unsigned int foff = 0, ftot = 0, aoff = 0, atot = 0; 
    for (int i=0; i!= theStDetConditions.nDet(); i++) {
      uint32_t detid = theStDetConditions.id(i);
      if (qualityFlags & BadModules) {
	bool isOn = quality->IsModuleUsable(detid);
	theStDetConditions.setActive(i,isOn);
	tot++; on += (unsigned int) isOn;
	if (qualityDebugFlags & BadModules) {
	  edm::LogInfo("MeasurementTracker")<< "MeasurementTrackerImpl::initializeStripStatus : detid " << detid << " is " << (isOn ?  "on" : "off");
	}
      } else {
	theStDetConditions.setActive(i,true);
      }
      // first turn all APVs and fibers ON
      theStDetConditions.set128StripStatus(i,true); 
      if (qualityFlags & BadAPVFibers) {
	short badApvs   = quality->getBadApvs(detid);
	short badFibers = quality->getBadFibers(detid);
	for (int j = 0; j < 6; j++) {
	  atot++;
	  if (badApvs & (1 << j)) {
	    theStDetConditions.set128StripStatus(i,false, j);
	    aoff++;
	  }
	}
	for (int j = 0; j < 3; j++) {
	  ftot++;
             if (badFibers & (1 << j)) {
	       theStDetConditions.set128StripStatus(i,false, 2*j);
	       theStDetConditions.set128StripStatus(i,false, 2*j+1);
	       foff++;
             }
	}
      } 
      auto & badStrips = theStDetConditions.getBadStripBlocks(i);
       badStrips.clear();
       if (qualityFlags & BadStrips) {
	 SiStripBadStrip::Range range = quality->getRange(detid);
	 for (SiStripBadStrip::ContainerIterator bit = range.first; bit != range.second; ++bit) {
	   badStrips.push_back(quality->decode(*bit));
	 }
       }
    }
    if (qualityDebugFlags & BadModules) {
      edm::LogInfo("MeasurementTracker StripModuleStatus") << 
	" Total modules: " << tot << ", active " << on <<", inactive " << (tot - on);
    }
    if (qualityDebugFlags & BadAPVFibers) {
      edm::LogInfo("MeasurementTracker StripAPVStatus") << 
	" Total APVs: " << atot << ", active " << (atot-aoff) <<", inactive " << (aoff);
        edm::LogInfo("MeasurementTracker StripFiberStatus") << 
	  " Total Fibers: " << ftot << ", active " << (ftot-foff) <<", inactive " << (foff);
    }
  } else {
    for (int i=0; i!=theStDetConditions.nDet(); i++) {
      theStDetConditions.setActive(i,true);          // module ON
      theStDetConditions.set128StripStatus(i,true);  // all APVs and fibers ON
    }
  }

}

void MeasurementTrackerImpl::initializePixelStatus(const SiPixelQuality *quality, const SiPixelFedCabling *pixelCabling, int qualityFlags, int qualityDebugFlags) {
  if ((quality != 0) && (qualityFlags != 0))  {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0, badrocs = 0; 
    for (std::vector<TkPixelMeasurementDet>::iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      uint32_t detid = ((*i).geomDet().geographicalId()).rawId();
      if (qualityFlags & BadModules) {
          bool isOn = quality->IsModuleUsable(detid);
          (i)->setActive(isOn);
          tot++; on += (unsigned int) isOn;
          if (qualityDebugFlags & BadModules) {
	    edm::LogInfo("MeasurementTracker")<< "MeasurementTrackerImpl::initializePixelStatus : detid " << detid << " is " << (isOn ?  "on" : "off");
          }
       } else {
          (i)->setActive(true);
       }
       if ((qualityFlags & BadROCs) && (quality->getBadRocs(detid) != 0)) {
          std::vector<LocalPoint> badROCs = quality->getBadRocPositions(detid, *theTrackerGeom, pixelCabling);
          badrocs += badROCs.size();
          (i)->setBadRocPositions(badROCs);
       } else {
          (i)->clearBadRocPositions();  
       }
    }
    if (qualityDebugFlags & BadModules) {
        edm::LogInfo("MeasurementTracker PixelModuleStatus") << 
            " Total modules: " << tot << ", active " << on <<", inactive " << (tot - on);
    }
    if (qualityDebugFlags & BadROCs) {
        edm::LogInfo("MeasurementTracker PixelROCStatus") << " Total of bad ROCs: " << badrocs ;
    }
  } else {
    for (std::vector<TkPixelMeasurementDet>::iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      (i)->setActive(true);          // module ON
    }
  }
}

