// -*- C++ -*-
//
// Package:    SiPixelMonitorRecHits
// Class:      SiPixelRecHitSource
// 
/**\class 

 Description: Pixel DQM source for RecHits

 Implementation:
     Originally based on the code for Digis, adapted
	to read RecHits and create relevant histograms
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelRecHitSource.cc,v 1.9 2008/06/26 08:36:23 merkelp Exp $
//
//
// Adapted by:  Keith Rose
//  	For use in SiPixelMonitorClient for RecHits
// Updated by: Lukas Wehrli
// for pixel offline DQM 

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitSource.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"


//
#include <string>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace edm;

SiPixelRecHitSource::SiPixelRecHitSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) ),
  saveFile( conf_.getUntrackedParameter<bool>("saveFile",false) ),
  isPIB( conf_.getUntrackedParameter<bool>("isPIB",false) ),
  slowDown( conf_.getUntrackedParameter<bool>("slowDown",false) ),
  modOn( conf_.getUntrackedParameter<bool>("modOn",true) ),
  ladOn( conf_.getUntrackedParameter<bool>("ladOn",false) ), 
  layOn( conf_.getUntrackedParameter<bool>("layOn",false) ), 
  phiOn( conf_.getUntrackedParameter<bool>("phiOn",false) ), 
  ringOn( conf_.getUntrackedParameter<bool>("ringOn",false) ), 
  bladeOn( conf_.getUntrackedParameter<bool>("bladeOn",false) ), 
  diskOn( conf_.getUntrackedParameter<bool>("diskOn",false) )
{
   theDMBE = edm::Service<DQMStore>().operator->();
   LogInfo ("PixelDQM") << "SiPixelRecHitSource::SiPixelRecHitSource: Got DQM BackEnd interface"<<endl;
}


SiPixelRecHitSource::~SiPixelRecHitSource()
{
   // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelRecHitSource::~SiPixelRecHitSource: Destructor"<<endl;
}


void SiPixelRecHitSource::beginJob(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelRecHitSource::beginJob - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
	    << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
	    << ringOn << std::endl;
  eventNo = 0;
	
  // Build map
  buildStructure(iSetup);
	
  // Book Monitoring Elements
  bookMEs();

}


void SiPixelRecHitSource::endJob(void){

  std::map<uint32_t,SiPixelRecHitModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    uint32_t TheID = (*struct_iter).first;
    int total = rechit_count[TheID];
    (*struct_iter).second->nfill(total, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn);
  }

  if(saveFile){
    LogInfo ("PixelDQM") << " SiPixelRecHitSource::endJob - Saving Root File " << std::endl;
    std::string outputFile = conf_.getParameter<std::string>("outputFile");
    theDMBE->save( outputFile );
  }

}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelRecHitSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  //cout << eventNo << endl;
  // get input data
  edm::Handle<SiPixelRecHitCollection>  recHitColl;
  iEvent.getByLabel( src_, recHitColl );

  std::map<uint32_t,SiPixelRecHitModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    uint32_t TheID = (*struct_iter).first;
	
    SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(TheID);
    SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
    
    SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
    SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;

      // if( pixelrechitRangeIteratorBegin == pixelrechitRangeIteratorEnd) {cout << "oops" << endl;}
      float rechit_x = 0;
      float rechit_y = 0;

  for ( ; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++) 
	{
	  

	  rechit_count[TheID]++;
	  //cout << TheID << endl;
	  SiPixelRecHit::ClusterRef const& clust = pixeliter->cluster();
	  int sizeX = (*clust).sizeX();
	  //cout << sizeX << endl;
	  int sizeY = (*clust).sizeY();
	  //cout << sizeY << endl;
	  LocalPoint lp = pixeliter->localPosition();
	  rechit_x = lp.x();
	  rechit_y = lp.y();
	  
	  LocalError lerr = pixeliter->localPositionError();
	  //  float lerr_x = sqrt(lerr.xx());
	  //  float lerr_y = sqrt(lerr.yy());
	  //cout << "hh" << endl;
	  (*struct_iter).second->fill(rechit_x, rechit_y, sizeX, sizeY,modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn);
	  //cout << "ii" << endl;
	
	}
    
  }

  // slow down...
  if(slowDown) usleep(10000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRecHitSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelRecHitSource::buildStructure" ;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){

      DetId detId = (*it)->geographicalId();
      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        if(isPIB) continue;
	LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRecHitModule* theModule = new SiPixelRecHitModule(id);
	thePixelStructure.insert(pair<uint32_t,SiPixelRecHitModule*> (id,theModule));
      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRecHitModule* theModule = new SiPixelRecHitModule(id);

        PixelEndcapName::HalfCylinder side = PixelEndcapName::PixelEndcapName(DetId::DetId(id)).halfCylinder();
        int disk   = PixelEndcapName::PixelEndcapName(DetId::DetId(id)).diskName();
        int blade  = PixelEndcapName::PixelEndcapName(DetId::DetId(id)).bladeName();
        int panel  = PixelEndcapName::PixelEndcapName(DetId::DetId(id)).pannelName();
        int module = PixelEndcapName::PixelEndcapName(DetId::DetId(id)).plaquetteName();

        char sside[80];  sprintf(sside,  "HalfCylinder_%i",side);
        char sdisk[80];  sprintf(sdisk,  "Disk_%i",disk);
        char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
        char spanel[80]; sprintf(spanel, "Panel_%i",panel);
        char smodule[80];sprintf(smodule,"Module_%i",module);
        std::string side_str = sside;
	std::string disk_str = sdisk;
	bool mask = side_str.find("HalfCylinder_1")!=string::npos||
	            side_str.find("HalfCylinder_2")!=string::npos||
	            side_str.find("HalfCylinder_4")!=string::npos||
	            disk_str.find("Disk_2")!=string::npos;
	if(isPIB && mask) continue;
	
	thePixelStructure.insert(pair<uint32_t,SiPixelRecHitModule*> (id,theModule));
      }
    }	    
  }

  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelRecHitSource::bookMEs(){
  
  std::map<uint32_t,SiPixelRecHitModule*>::iterator struct_iter;
  theDMBE->setVerbose(0);
    
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
	(*struct_iter).second->book( conf_ );
      } else {
	if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1)){
	(*struct_iter).second->book( conf_,1);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if(layOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,2)){
	(*struct_iter).second->book( conf_,2);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if(phiOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,3)){
	(*struct_iter).second->book( conf_,3);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4)){
	(*struct_iter).second->book( conf_,4);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,5)){
	(*struct_iter).second->book( conf_,5);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,6)){
	(*struct_iter).second->book( conf_,6);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelRecHitSource);
