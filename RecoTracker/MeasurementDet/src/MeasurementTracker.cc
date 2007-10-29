#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

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
				       /*const SiStripDetCabling *stripCabling,*/
				       const SiStripNoises *stripNoises,
				       bool isRegional) :
  pset_(conf),lastEventNumberPixels(0),lastEventNumberStrips(0),
  lastRunNumberPixels(0),lastRunNumberStrips(0),
  thePixelCPE(pixelCPE),theStripCPE(stripCPE),theHitMatcher(hitMatcher),
  theTrackerGeom(trackerGeom),theGeometricSearchTracker(geometricSearchTracker)
  ,dummyStripNoises(0), isRegional_(isRegional)
{
  this->initialize();
  this->initializeStripStatus(stripQuality);
  this->initializeStripNoises(stripNoises);
}

MeasurementTracker::~MeasurementTracker()
{
  if (dummyStripNoises) delete dummyStripNoises;

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
  if( (lastEventNumberPixels == event.id().event()) && 
      (lastRunNumberPixels   == event.id().run() )     ) return;
  
  lastEventNumberPixels = event.id().event();
  lastRunNumberPixels = event.id().run();

  typedef edm::DetSetVector<SiPixelCluster> ::detset   PixelDetSet;

  // Pixel Clusters
  std::string pixelClusterProducer = pset_.getParameter<std::string>("pixelClusterProducer");
  if( !pixelClusterProducer.compare("") ) { //clusters have not been produced
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {
      (**i).setEmpty();
    }
  }else{  
    edm::Handle<edm::DetSetVector<SiPixelCluster> > pixelClusters;
    event.getByLabel(pixelClusterProducer, pixelClusters);
    const  edm::DetSetVector<SiPixelCluster>* pixelCollection = pixelClusters.product();
    
    for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
	 i!=thePixelDets.end(); i++) {

      // foreach det get cluster range
      unsigned int id = (**i).geomDet().geographicalId().rawId();
      edm::DetSetVector<SiPixelCluster>::const_iterator it = pixelCollection->find( id );
      if ( it != pixelCollection->end() ){            
	const PixelDetSet & detSet = (*pixelCollection)[ id ];
	// push cluster range in det
	(**i).update( detSet, pixelClusters, id );
      }else{
	(**i).setEmpty();
      }
    }
  }
  
}

void MeasurementTracker::updateStrips( const edm::Event& event) const
{
  // avoid to update twice from the same event
  if( (lastEventNumberStrips == event.id().event()) && 
      (lastRunNumberStrips   == event.id().run() )     ) return;
  
  lastEventNumberStrips = event.id().event();
  lastRunNumberStrips = event.id().run();


  typedef edm::DetSetVector<SiStripCluster> ::detset   StripDetSet;

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
      edm::Handle<edm::DetSetVector<SiStripCluster> > clusterHandle;
      event.getByLabel(stripClusterProducer, clusterHandle);
      const edm::DetSetVector<SiStripCluster>* clusterCollection = clusterHandle.product();

      for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	   i!=theStripDets.end(); i++) {
	
	// foreach det get cluster range
	unsigned int id = (**i).geomDet().geographicalId().rawId();
	edm::DetSetVector<SiStripCluster>::const_iterator it = clusterCollection->find( id );
	if ( it != clusterCollection->end() ){
	  const StripDetSet & detSet = (*clusterCollection)[ id ];
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
      edm::Handle<edm::SiStripRefGetter<SiStripCluster> > refClusterHandle;
      event.getByLabel(stripClusterProducer, refClusterHandle);
      
      uint32_t tmpId=0;
      vector<SiStripCluster>::const_iterator beginIterator;
      edm::SiStripRefGetter<SiStripCluster>::const_iterator iregion = refClusterHandle->begin();
      for(;iregion!=refClusterHandle->end();++iregion) {
	vector<SiStripCluster>::const_iterator icluster = iregion->begin();
	tmpId = icluster->geographicalId();
	beginIterator = icluster;

	//std::cout << "== tmpId ad inizio loop dentro region: " << tmpId << std::endl;

	for (;icluster!=iregion->end();icluster++) {
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
	    theConcreteDetUpdatable->update(beginIterator,icluster,refClusterHandle,tmpId);
	    //cannot we avoid to update the det with detId of itself??

	    tmpId = icluster->geographicalId();
	    beginIterator = icluster;
	    if( icluster == (iregion->end()-1)){
	      const TkStripMeasurementDet* theConcreteDet = 
	      dynamic_cast<const TkStripMeasurementDet*>(idToDet(DetId(tmpId)));
	      
	      if(theConcreteDet == 0)
	      throw MeasurementDetException("failed casting to TkStripMeasurementDet*");	    
	      TkStripMeasurementDet*  theConcreteDetUpdatable = 
	      const_cast<TkStripMeasurementDet*>(theConcreteDet);
	      theConcreteDetUpdatable->update(icluster,iregion->end(),refClusterHandle,tmpId);
	    }	 
	  }else if( icluster == (iregion->end()-1)){	   
	    const TkStripMeasurementDet* theConcreteDet = 
	      dynamic_cast<const TkStripMeasurementDet*>(idToDet(DetId(tmpId)));
	    
	    if(theConcreteDet == 0)
	      throw MeasurementDetException("failed casting to TkStripMeasurementDet*");	    
	    TkStripMeasurementDet*  theConcreteDetUpdatable = 
	      const_cast<TkStripMeasurementDet*>(theConcreteDet);
	    //std::cout << "=== option3. fill det with id,#clust: " << tmpId  << " , " 
	    //      << iregion->end() - beginIterator << std::endl;
	    theConcreteDetUpdatable->update(beginIterator,iregion->end(),refClusterHandle,tmpId);	 
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

void MeasurementTracker::initializeStripNoises(const SiStripNoises *noises) const {
    if (noises) {
        for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
                i!=theStripDets.end(); i++) {
            uint32_t detid = ((**i).geomDet().geographicalId()).rawId();
            (**i).setNoises(noises->getRange(detid));
        } 
    } else {
        dummyStripNoises = new SiStripNoises();
        for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
                i!=theStripDets.end(); i++) {
            (**i).setNoises(dummyStripNoises->getRange(0));
        } 
    }                                                                                         
}                                       

void MeasurementTracker::initializeStripStatus(const SiStripQuality *quality) const {
  if (quality)  {
    //std::pair<double,double> t1,t2,t3; 
    //TimeMe timer("[*GIO*] MTuSS TimeMe",false); t1 = timer.lap();
    unsigned int on = 0, tot = 0; 
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      uint32_t detid = ((**i).geomDet().geographicalId()).rawId();
      bool isOn = ! quality->IsModuleBad(detid);
      (*i)->setActive(isOn);
      tot++; on += (unsigned int) isOn;
    }
    //t2 = timer.lap();
    //edm::LogInfo("[*GIO*] MTuSS") << "It took " << (t2.first - t1.first) << " s (total) to dispatch " << tot << " modules";
    std::cout << ("[*GIO*] MTuSS") << " Total modules: " << tot << ", active " << on <<", inactive " << (tot - on) << std::endl;
  } else {
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      (*i)->setActive(true);
    }
  }
}
/* LEGACY METHOD - Deprecated
void MeasurementTracker::initializeStripStatus(const SiStripDetCabling *cabling) const {
  if (cabling)  {
    //std::pair<double,double> t1,t2,t3; 
    //TimeMe timer("[*GIO*] MTuSS TimeMe",false); t1 = timer.lap();

    const std::map< uint32_t, std::vector<FedChannelConnection> > & activeModules = cabling->getDetCabling ();
    
    //t3 = timer.lap();
    //unsigned int on = 0, tot = 0; 
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      uint32_t detid = ((**i).geomDet().geographicalId()).rawId();
      std::map< uint32_t, std::vector<FedChannelConnection> >::const_iterator it =  activeModules.find(detid);
      bool isOn = (it!=activeModules.end());
      (*i)->setActive(isOn);
      //tot++; on += (unsigned int) isOn;
    }
    
    //t2 = timer.lap();
    //edm::LogInfo("[*GIO*] MTuSS") << "It took " << (t2.first - t1.first) << " s (total) to dispatch " << activeModules.size() << " modules (" 
    //          << (t3.first - t1.first) << " in getActiveDetectorRawIds and " << (t2.first - t3.first) << " in searching";
    //edm::LogInfo("[*GIO*] MTuSS") << " Total modules: " << tot << ", active " << on <<", inactive " << (tot -on);
    
  } else {
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      (*i)->setActive(true);
    }
  }
}
*/
