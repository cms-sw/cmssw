#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

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

#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/CPEFromDetPosition.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"  

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkPixelMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkGluedMeasurementDet.h"

#include "Utilities/Timing/interface/TimingReport.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/UpdaterService.h"

#include <iostream>
#include <typeinfo>
#include <map>

using namespace std;

MeasurementTracker::MeasurementTracker(const edm::ParameterSet&              conf,
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
				       bool isRegional) :
  pset_(conf),
  name_(conf.getParameter<std::string>("ComponentName")),
  thePixelCPE(pixelCPE),theStripCPE(stripCPE),theHitMatcher(hitMatcher),
  theTrackerGeom(trackerGeom),theGeometricSearchTracker(geometricSearchTracker),
  theInactivePixelDetectorLabels(conf.getParameter<std::vector<edm::InputTag> >("inactivePixelDetectorLabels")),
  theInactiveStripDetectorLabels(conf.getParameter<std::vector<edm::InputTag> >("inactiveStripDetectorLabels")),
  isRegional_(isRegional)
{
  this->initialize();
  this->initializeStripStatus(stripQuality, stripQualityFlags, stripQualityDebugFlags);
  this->initializePixelStatus(pixelQuality, pixelCabling, pixelQualityFlags, pixelQualityDebugFlags);
}

MeasurementTracker::~MeasurementTracker()
{
  for(vector<TkPixelMeasurementDet*>::const_iterator it=thePixelDets.begin(); it!=thePixelDets.end(); ++it){
    delete *it;
  }

  for(vector<TkStripMeasurementDet*>::const_iterator it=theStripDets.begin(); it!=theStripDets.end(); ++it){
    delete *it;
  }

  for(vector<TkGluedMeasurementDet*>::const_iterator it=theGluedDets.begin(); it!=theGluedDets.end(); ++it){
    delete *it;
  }
}


void MeasurementTracker::initialize() const
{  
  addPixelDets( theTrackerGeom->detsPXB());
  addPixelDets( theTrackerGeom->detsPXF());
  addStripDets( theTrackerGeom->detsTIB());
  addStripDets( theTrackerGeom->detsTID());
  addStripDets( theTrackerGeom->detsTOB());
  addStripDets( theTrackerGeom->detsTEC());  
}


void MeasurementTracker::addPixelDets( const TrackingGeometry::DetContainer& dets) const
{
  for (TrackerGeometry::DetContainer::const_iterator gd=dets.begin();
       gd != dets.end(); gd++) {
    addPixelDet(*gd, thePixelCPE);
  }  
}

void MeasurementTracker::addStripDets( const TrackingGeometry::DetContainer& dets) const
{
  for (TrackerGeometry::DetContainer::const_iterator gd=dets.begin();
       gd != dets.end(); gd++) {

    const GeomDetUnit* gdu = dynamic_cast<const GeomDetUnit*>(*gd);

    //    StripSubdetector stripId( (**gd).geographicalId());
    //     bool isDetUnit( gdu != 0);
    //     cout << "StripSubdetector glued? " << stripId.glued() 
    // 	 << " is DetUnit? " << isDetUnit << endl;

    if (gdu != 0) {
      addStripDet(*gd, theStripCPE);
    }
    else {
      const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*>(*gd);
      if (gluedDet == 0) {
	throw MeasurementDetException("MeasurementTracker ERROR: GeomDet neither DetUnit nor GluedDet");
      }
      addGluedDet(gluedDet, theHitMatcher);
    }  
  }
}

void MeasurementTracker::addStripDet( const GeomDet* gd,
				      const StripClusterParameterEstimator* cpe) const
{
  try {
    TkStripMeasurementDet* det = new TkStripMeasurementDet( gd, cpe,isRegional_);
    theStripDets.push_back(det);
    theDetMap[gd->geographicalId()] = det;
  }
  catch(MeasurementDetException& err){
    edm::LogError("MeasurementDet") << "Oops, got a MeasurementDetException: " << err.what() ;
  }
}

void MeasurementTracker::addPixelDet( const GeomDet* gd,
				      const PixelClusterParameterEstimator* cpe) const
{
  TkPixelMeasurementDet* det = new TkPixelMeasurementDet( gd, cpe);
  thePixelDets.push_back(det);
  theDetMap[gd->geographicalId()] = det;
}

void MeasurementTracker::addGluedDet( const GluedGeomDet* gd,
				      const SiStripRecHitMatcher* matcher) const
{
  const MeasurementDet* monoDet = idToDet( gd->monoDet()->geographicalId());
  if (monoDet == 0) {
    addStripDet(gd->monoDet(), theStripCPE);  // in case glued det comes before components
    monoDet = idToDet( gd->monoDet()->geographicalId());
  }

  const MeasurementDet* stereoDet = idToDet( gd->stereoDet()->geographicalId());
  if (stereoDet == 0) {
    addStripDet(gd->stereoDet(), theStripCPE);  // in case glued det comes before components
    stereoDet = idToDet( gd->stereoDet()->geographicalId());
  }

  if (monoDet == 0 || stereoDet == 0) {
    edm::LogError("MeasurementDet") << "MeasurementTracker ERROR: GluedDet components not found as MeasurementDets ";
    throw MeasurementDetException("MeasurementTracker ERROR: GluedDet components not found as MeasurementDets");
  }

  TkGluedMeasurementDet* det = new TkGluedMeasurementDet( gd, theHitMatcher,
							  monoDet, stereoDet);
  theGluedDets.push_back( det);
  theDetMap[gd->geographicalId()] = det;
}

void MeasurementTracker::update( const edm::Event& event) const
{
  updatePixels(event);
  updateStrips(event);
  
  /*
  for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
       i!=theStripDets.end(); i++) {
    if( (*i)->isEmpty()){
      std::cout << "stripDet id, #hits: " 
		<<  (*i)->geomDet().geographicalId().rawId() << " , "
		<< 0 << std::endl;
    }else{
      std::cout << "stripDet id, #hits: " 
		<<  (*i)->geomDet().geographicalId().rawId() << " , "
		<< (*i)->size() << std::endl;
    }
  }
  */
}


void MeasurementTracker::updatePixels( const edm::Event& event) const
{
  // avoid to update twice from the same event
  if (!edm::Service<UpdaterService>()->checkOnce("MeasurementTracker::updatePixels::"+name_)) return;

  typedef edmNew::DetSet<SiPixelCluster> PixelDetSet;

  bool switchOffPixelsIfEmpty = (!pset_.existsAs<bool>("switchOffPixelsIfEmpty")) ||
                                (pset_.getParameter<bool>("switchOffPixelsIfEmpty"));

  std::vector<uint32_t> rawInactiveDetIds; 
  if (!theInactivePixelDetectorLabels.empty()) {
    edm::Handle<DetIdCollection> detIds;
    for (std::vector<edm::InputTag>::const_iterator itt = theInactivePixelDetectorLabels.begin(), edt = theInactivePixelDetectorLabels.end(); 
            itt != edt; ++itt) {
        event.getByLabel(*itt, detIds);
        rawInactiveDetIds.insert(rawInactiveDetIds.end(), detIds->begin(), detIds->end());
    }
    if (!rawInactiveDetIds.empty()) std::sort(rawInactiveDetIds.begin(), rawInactiveDetIds.end());
  }
  // Pixel Clusters
  std::string pixelClusterProducer = pset_.getParameter<std::string>("pixelClusterProducer");
  if( pixelClusterProducer.empty() ) { //clusters have not been produced
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      if (switchOffPixelsIfEmpty) {
        (**i).setActiveThisEvent(false);
      }else{
	(**i).setEmpty();
      }
    }
  }else{  
    edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
    event.getByLabel(pixelClusterProducer, pixelClusters);
    const  edmNew::DetSetVector<SiPixelCluster>* pixelCollection = pixelClusters.product();
   
    if (switchOffPixelsIfEmpty && pixelCollection->empty()) {
        for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
             i!=thePixelDets.end(); i++) {
              (**i).setActiveThisEvent(false);
        }
    } else { 
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {

      // foreach det get cluster range
      unsigned int id = (**i).geomDet().geographicalId().rawId();
      if (!rawInactiveDetIds.empty() && std::binary_search(rawInactiveDetIds.begin(), rawInactiveDetIds.end(), id)) {
        (**i).setActiveThisEvent(false); continue;
      }
      edmNew::DetSetVector<SiPixelCluster>::const_iterator it = pixelCollection->find( id );
      if ( it != pixelCollection->end() ){            
	// push cluster range in det
	(**i).update( *it, pixelClusters, id );
      }else{
	(**i).setEmpty();
      }
    }
    }
  }
  
}

void MeasurementTracker::updateStrips( const edm::Event& event) const
{
  // avoid to update twice from the same event
  if (!edm::Service<UpdaterService>()->checkOnce("MeasurementTracker::updateStrips::"+name_)) return;

  typedef edmNew::DetSet<SiStripCluster>   StripDetSet;

  std::vector<uint32_t> rawInactiveDetIds;
  if (!theInactiveStripDetectorLabels.empty()) {
    edm::Handle<DetIdCollection> detIds;
    for (std::vector<edm::InputTag>::const_iterator itt = theInactiveStripDetectorLabels.begin(), edt = theInactiveStripDetectorLabels.end(); 
            itt != edt; ++itt) {
        event.getByLabel(*itt, detIds);
        rawInactiveDetIds.insert(rawInactiveDetIds.end(), detIds->begin(), detIds->end());
    }
    if (!rawInactiveDetIds.empty()) std::sort(rawInactiveDetIds.begin(), rawInactiveDetIds.end());
  }

  // Strip Clusters
  std::string stripClusterProducer = pset_.getParameter<std::string>("stripClusterProducer");
  if( !stripClusterProducer.compare("") ) { //clusters have not been produced
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      (**i).setEmpty();
    }
  }else{
    //=========  actually load cluster =============
    if(!isRegional_){
      edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
      event.getByLabel(stripClusterProducer, clusterHandle);
      const edmNew::DetSetVector<SiStripCluster>* clusterCollection = clusterHandle.product();

      for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	   i!=theStripDets.end(); i++) {
	
	// foreach det get cluster range
	unsigned int id = (**i).geomDet().geographicalId().rawId();
        if (!rawInactiveDetIds.empty() && std::binary_search(rawInactiveDetIds.begin(), rawInactiveDetIds.end(), id)) {
            (**i).setActiveThisEvent(false); continue;
        }
	edmNew::DetSetVector<SiStripCluster>::const_iterator it = clusterCollection->find( id );
	if ( it != clusterCollection->end() ){
	  StripDetSet detSet = (*clusterCollection)[ id ];
	  // push cluster range in det
	  (**i).update( detSet, clusterHandle, id );
	  
	}else{
	  (**i).setEmpty();
	}
      }
    }else{
      //first clear all of them
      for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	   i!=theStripDets.end(); i++) {
	(**i).setEmpty();
      }
            
      //then set the not-empty ones only
      edm::Handle<edm::RefGetter<SiStripCluster> > refClusterHandle;
      event.getByLabel(stripClusterProducer, refClusterHandle);
      
      std::string lazyGetter = pset_.getParameter<std::string>("stripLazyGetterProducer");
      edm::Handle<edm::LazyGetter<SiStripCluster> > lazyClusterHandle;
      event.getByLabel(lazyGetter,lazyClusterHandle);

      uint32_t tmpId=0;
      vector<SiStripCluster>::const_iterator beginIterator;
      edm::RefGetter<SiStripCluster>::const_iterator iregion = refClusterHandle->begin();
      for(;iregion!=refClusterHandle->end();++iregion) {
	const edm::RegionIndex<SiStripCluster>& region = *iregion;
	vector<SiStripCluster>::const_iterator icluster = region.begin();
	const vector<SiStripCluster>::const_iterator endIterator = region.end();
	tmpId = icluster->geographicalId();
	beginIterator = icluster;

	//std::cout << "== tmpId ad inizio loop dentro region: " << tmpId << std::endl;

	for (;icluster!=endIterator;icluster++) {
	  //std::cout << "===== cluster id,pos " 
	  //  << icluster->geographicalId() << " , " << icluster->barycenter()
	  //  << std::endl;
	  //std::cout << "=====making ref in recHits() " << std::endl;
	  if( icluster->geographicalId() != tmpId){ 
	    //std::cout << "geo!=tmpId" << std::endl;
	    //we should find a way to avoid this casting. it is slow
	    //create also another map for TkStripMeasurementDet ??

	    // the following castings are really ugly. To be corrected ASAP
	    const TkStripMeasurementDet* theConcreteDet = 
	      dynamic_cast<const TkStripMeasurementDet*>(idToDet(DetId(tmpId)));
	    
	    if(theConcreteDet == 0)
	      throw MeasurementDetException("failed casting to TkStripMeasurementDet*");	    
	    TkStripMeasurementDet*  theConcreteDetUpdatable = 
	      const_cast<TkStripMeasurementDet*>(theConcreteDet);
	    theConcreteDetUpdatable->update(beginIterator,icluster,lazyClusterHandle,tmpId);
	    //cannot we avoid to update the det with detId of itself??

	    tmpId = icluster->geographicalId();
	    beginIterator = icluster;
	    if( icluster == (endIterator-1)){
	      const TkStripMeasurementDet* theConcreteDet = 
	      dynamic_cast<const TkStripMeasurementDet*>(idToDet(DetId(tmpId)));
	      
	      if(theConcreteDet == 0)
	      throw MeasurementDetException("failed casting to TkStripMeasurementDet*");	    
	      TkStripMeasurementDet*  theConcreteDetUpdatable = 
	      const_cast<TkStripMeasurementDet*>(theConcreteDet);
	      theConcreteDetUpdatable->update(icluster,endIterator,lazyClusterHandle,tmpId);
	    }	 
	  }else if( icluster == (endIterator-1)){	   
	    const TkStripMeasurementDet* theConcreteDet = 
	      dynamic_cast<const TkStripMeasurementDet*>(idToDet(DetId(tmpId)));
	    
	    if(theConcreteDet == 0)
	      throw MeasurementDetException("failed casting to TkStripMeasurementDet*");	    
	    TkStripMeasurementDet*  theConcreteDetUpdatable = 
	      const_cast<TkStripMeasurementDet*>(theConcreteDet);
	    //std::cout << "=== option3. fill det with id,#clust: " << tmpId  << " , " 
	    //      << iregion->end() - beginIterator << std::endl;
	    theConcreteDetUpdatable->update(beginIterator,endIterator,lazyClusterHandle,tmpId);	 
	  }	  
	}//end loop cluster in one ragion
      }
    }//end of block for updating with regional clusters 
  }

}



const MeasurementDet* 
MeasurementTracker::idToDet(const DetId& id) const
{
  std::map<DetId,MeasurementDet*>::const_iterator it = theDetMap.find(id);
  if(it !=theDetMap.end()) {
    return it->second;
  }else{
    //throw exception;
  }
  
  return 0; //to avoid compile warning
}

void MeasurementTracker::initializeStripStatus(const SiStripQuality *quality, int qualityFlags, int qualityDebugFlags) const {
  TkStripMeasurementDet::BadStripCuts badStripCuts[4];
  if (qualityFlags & BadStrips) {
     edm::ParameterSet cutPset = pset_.getParameter<edm::ParameterSet>("badStripCuts");
     badStripCuts[SiStripDetId::TIB-3] = TkStripMeasurementDet::BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TIB"));
     badStripCuts[SiStripDetId::TOB-3] = TkStripMeasurementDet::BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TOB"));
     badStripCuts[SiStripDetId::TID-3] = TkStripMeasurementDet::BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TID"));
     badStripCuts[SiStripDetId::TEC-3] = TkStripMeasurementDet::BadStripCuts(cutPset.getParameter<edm::ParameterSet>("TEC"));
  }

  if ((quality != 0) && (qualityFlags != 0))  {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0; 
    unsigned int foff = 0, ftot = 0, aoff = 0, atot = 0; 
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      uint32_t detid = ((**i).geomDet().geographicalId()).rawId();
      if (qualityFlags & BadModules) {
          bool isOn = quality->IsModuleUsable(detid);
          (*i)->setActive(isOn);
          tot++; on += (unsigned int) isOn;
          if (qualityDebugFlags & BadModules) {
	    edm::LogInfo("MeasurementTracker")<< "MeasurementTracker::initializeStripStatus : detid " << detid << " is " << (isOn ?  "on" : "off");
          }
       } else {
          (*i)->setActive(true);
       }
       // first turn all APVs and fibers ON
       (*i)->set128StripStatus(true); 
       if (qualityFlags & BadAPVFibers) {
          short badApvs   = quality->getBadApvs(detid);
          short badFibers = quality->getBadFibers(detid);
          for (int j = 0; j < 6; j++) {
             atot++;
             if (badApvs & (1 << j)) {
                (*i)->set128StripStatus(false, j);
                aoff++;
             }
          }
          for (int j = 0; j < 3; j++) {
             ftot++;
             if (badFibers & (1 << j)) {
                (*i)->set128StripStatus(false, 2*j);
                (*i)->set128StripStatus(false, 2*j+1);
                foff++;
             }
          }
          (*i)->setMaskBad128StripBlocks(qualityFlags & MaskBad128StripBlocks != 0);
       } 
       std::vector<TkStripMeasurementDet::BadStripBlock> &badStrips = (*i)->getBadStripBlocks();
       badStrips.clear();
       if (qualityFlags & BadStrips) {
            SiStripBadStrip::Range range = quality->getRange(detid);
            for (SiStripBadStrip::ContainerIterator bit = range.first; bit != range.second; ++bit) {
                badStrips.push_back(quality->decode(*bit));
            }
            (*i)->setBadStripCuts(badStripCuts[SiStripDetId(detid).subdetId()-3]);
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
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      (*i)->setActive(true);          // module ON
      (*i)->set128StripStatus(true);  // all APVs and fibers ON
    }
  }
}

void MeasurementTracker::initializePixelStatus(const SiPixelQuality *quality, const SiPixelFedCabling *pixelCabling, int qualityFlags, int qualityDebugFlags) const {
  if ((quality != 0) && (qualityFlags != 0))  {
    edm::LogInfo("MeasurementTracker") << "qualityFlags = " << qualityFlags;
    unsigned int on = 0, tot = 0, badrocs = 0; 
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      uint32_t detid = ((**i).geomDet().geographicalId()).rawId();
      if (qualityFlags & BadModules) {
          bool isOn = quality->IsModuleUsable(detid);
          (*i)->setActive(isOn);
          tot++; on += (unsigned int) isOn;
          if (qualityDebugFlags & BadModules) {
	    edm::LogInfo("MeasurementTracker")<< "MeasurementTracker::initializePixelStatus : detid " << detid << " is " << (isOn ?  "on" : "off");
          }
       } else {
          (*i)->setActive(true);
       }
       if ((qualityFlags & BadROCs) && (quality->getBadRocs(detid) != 0)) {
          std::vector<LocalPoint> badROCs = quality->getBadRocPositions(detid, *theTrackerGeom, pixelCabling);
          badrocs += badROCs.size();
          (*i)->setBadRocPositions(badROCs);
       } else {
          (*i)->clearBadRocPositions();  
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
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      (*i)->setActive(true);          // module ON
    }
  }
}

