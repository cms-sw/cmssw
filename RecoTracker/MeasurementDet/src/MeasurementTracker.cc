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

MeasurementTracker::MeasurementTracker(const edm::ParameterSet&              conf,
				       const PixelClusterParameterEstimator* pixelCPE,
				       const StripClusterParameterEstimator* stripCPE,
				       const SiStripRecHitMatcher*  hitMatcher,
				       const TrackerGeometry*  trackerGeom,
				       const GeometricSearchTracker* geometricSearchTracker,
				       const SiStripDetCabling *stripCabling,
				       const SiStripNoises *stripNoises) :
  pset_(conf),lastEventNumber(0),lastRunNumber(0),
  thePixelCPE(pixelCPE),theStripCPE(stripCPE),theHitMatcher(hitMatcher),
  theTrackerGeom(trackerGeom),theGeometricSearchTracker(geometricSearchTracker)
  ,dummyStripNoises(0)
{
  this->initialize();
  this->initializeStripStatus(stripCabling);
  this->initializeStripNoises(stripNoises);
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
    TkStripMeasurementDet* det = new TkStripMeasurementDet( gd, cpe);
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
  // avoid to update twice from the same event
  if( (lastEventNumber == event.id().event()) && 
      (lastRunNumber   == event.id().run() )     ) return;
  
  lastEventNumber = event.id().event();
  lastRunNumber = event.id().run();

  typedef edm::DetSetVector<SiStripCluster> ::detset   StripDetSet;
  typedef edm::DetSetVector<SiPixelCluster> ::detset   PixelDetSet;

  // Strip Clusters
  std::string stripClusterProducer = pset_.getParameter<std::string>("stripClusterProducer");
  if( !stripClusterProducer.compare("") ) { //clusters have not been produced
    for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
	 i!=theStripDets.end(); i++) {
      (**i).setEmpty();
    }
  }else{  
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
  }


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

  /// or maybe faster: loop over all strip dets and clear them
  /// loop over dets with clusters and set range

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

