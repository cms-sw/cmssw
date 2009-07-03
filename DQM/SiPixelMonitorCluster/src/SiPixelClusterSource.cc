// -*- C++ -*-
//
// Package:    SiPixelMonitorCluster
// Class:      SiPixelClusterSource
// 
/**\class 

 Description: Pixel DQM source for Clusters

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
// $Id: SiPixelClusterSource.cc,v 1.16 2009/06/18 10:27:42 zablocki Exp $
//
//
// Updated by: Lukas Wehrli
// for pixel offline DQM 
#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterSource.h"
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
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
//
#include <string>
#include <stdlib.h>

using namespace std;
using namespace edm;

SiPixelClusterSource::SiPixelClusterSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) ),
  saveFile( conf_.getUntrackedParameter<bool>("saveFile",false) ),
  isPIB( conf_.getUntrackedParameter<bool>("isPIB",false) ),
  slowDown( conf_.getUntrackedParameter<bool>("slowDown",false) ),
  modOn( conf_.getUntrackedParameter<bool>("modOn",true) ),
  twoDimOn( conf_.getUntrackedParameter<bool>("twoDimOn",true) ),
  reducedSet( conf_.getUntrackedParameter<bool>("reducedSet",false) ),
  ladOn( conf_.getUntrackedParameter<bool>("ladOn",false) ), 
  layOn( conf_.getUntrackedParameter<bool>("layOn",false) ), 
  phiOn( conf_.getUntrackedParameter<bool>("phiOn",false) ), 
  ringOn( conf_.getUntrackedParameter<bool>("ringOn",false) ), 
  bladeOn( conf_.getUntrackedParameter<bool>("bladeOn",false) ), 
  diskOn( conf_.getUntrackedParameter<bool>("diskOn",false) )
{
   theDMBE = edm::Service<DQMStore>().operator->();
   LogInfo ("PixelDQM") << "SiPixelClusterSource::SiPixelClusterSource: Got DQM BackEnd interface"<<endl;
}


SiPixelClusterSource::~SiPixelClusterSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelClusterSource::~SiPixelClusterSource: Destructor"<<endl;
}


void SiPixelClusterSource::beginJob(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelClusterSource::beginJob - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
	    << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
	    << ringOn << std::endl;
  LogInfo ("PixelDQM") << "2DIM IS " << twoDimOn << "\n";

  eventNo = 0;
  // Build map
  buildStructure(iSetup);
  // Book Monitoring Elements
  bookMEs();

}


void SiPixelClusterSource::endJob(void){
  if(saveFile){
    LogInfo ("PixelDQM") << " SiPixelClusterSource::endJob - Saving Root File " << std::endl;
    std::string outputFile = conf_.getParameter<std::string>("outputFile");
    theDMBE->save( outputFile.c_str() );
  }
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelClusterSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;

  // get input data
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  input;
  iEvent.getByLabel( src_, input );

  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    (*struct_iter).second->fill(*input, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn, twoDimOn, reducedSet);
    
  }

  // slow down...
  if(slowDown) usleep(10000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelClusterSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelClusterSource::buildStructure" ;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){

      DetId detId = (*it)->geographicalId();
      const GeomDetUnit      * geoUnit = pDD->idToDetUnit( detId );
      const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      int nrows = (pixDet->specificTopology()).nrows();
      int ncols = (pixDet->specificTopology()).ncolumns();

      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        if(isPIB) continue;
	LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelClusterModule* theModule = new SiPixelClusterModule(id, ncols, nrows);
	thePixelStructure.insert(pair<uint32_t,SiPixelClusterModule*> (id,theModule));

      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelClusterModule* theModule = new SiPixelClusterModule(id, ncols, nrows);

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
	// clutch to take all of FPIX, but no BPIX:
	mask = false;
	if(isPIB && mask) continue;
	
	thePixelStructure.insert(pair<uint32_t,SiPixelClusterModule*> (id,theModule));
      }

    }
  }
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelClusterSource::bookMEs(){
  
  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
    
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
        (*struct_iter).second->book( conf_,0,twoDimOn,reducedSet);
      } else {
        
        if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelClusterSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1)){
	(*struct_iter).second->book( conf_,1,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if(layOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,2)){
	(*struct_iter).second->book( conf_,2,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if(phiOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,3)){
	(*struct_iter).second->book( conf_,3,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4)){
	(*struct_iter).second->book( conf_,4,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,5)){
	(*struct_iter).second->book( conf_,5,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,6)){
	(*struct_iter).second->book( conf_,6,twoDimOn,reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelClusterSource);
