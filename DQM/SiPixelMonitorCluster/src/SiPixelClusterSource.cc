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
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
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
  diskOn( conf_.getUntrackedParameter<bool>("diskOn",false) ),
  smileyOn(conf_.getUntrackedParameter<bool>("smileyOn",false) ),
  bigEventSize( conf_.getUntrackedParameter<int>("bigEventSize",100) ), 
  isUpgrade( conf_.getUntrackedParameter<bool>("isUpgrade",false) )
{
   theDMBE = edm::Service<DQMStore>().operator->();
   LogInfo ("PixelDQM") << "SiPixelClusterSource::SiPixelClusterSource: Got DQM BackEnd interface"<<endl;

   //set Token(-s)
   srcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(conf_.getParameter<edm::InputTag>("src"));
}


SiPixelClusterSource::~SiPixelClusterSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelClusterSource::~SiPixelClusterSource: Destructor"<<endl;

  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++){
    delete struct_iter->second;
    struct_iter->second = 0;
  }
}


void SiPixelClusterSource::beginJob(){
  firstRun = true;
}

void SiPixelClusterSource::beginRun(const edm::Run& r, const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelClusterSource::beginJob - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
	    << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
	    << ringOn << std::endl;
  LogInfo ("PixelDQM") << "2DIM IS " << twoDimOn << "\n";
  LogInfo ("PixelDQM") << "Smiley (Cluster sizeY vs. Cluster eta) is " << smileyOn << "\n";

  if(firstRun){
    eventNo = 0;
    lumSec = 0;
    nLumiSecs = 0;
    nBigEvents = 0;
    // Build map
    buildStructure(iSetup);
    // Book Monitoring Elements
    bookMEs();
    // Book occupancy maps in global coordinates for all clusters:
    theDMBE->setCurrentFolder("Pixel/Clusters/OffTrack");
    //bpix
    meClPosLayer1 = theDMBE->book2D("position_siPixelClusters_Layer_1","Clusters Layer1;Global Z (cm);Global #phi",200,-30.,30.,128,-3.2,3.2);
    meClPosLayer2 = theDMBE->book2D("position_siPixelClusters_Layer_2","Clusters Layer2;Global Z (cm);Global #phi",200,-30.,30.,128,-3.2,3.2);
    meClPosLayer3 = theDMBE->book2D("position_siPixelClusters_Layer_3","Clusters Layer3;Global Z (cm);Global #phi",200,-30.,30.,128,-3.2,3.2);
    if (isUpgrade) {
      meClPosLayer4 = theDMBE->book2D("position_siPixelClusters_Layer_4","Clusters Layer4;Global Z (cm);Global #phi",200,-30.,30.,128,-3.2,3.2);
    }
    //fpix
    meClPosDisk1pz = theDMBE->book2D("position_siPixelClusters_pz_Disk_1","Clusters +Z Disk1;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
    meClPosDisk2pz = theDMBE->book2D("position_siPixelClusters_pz_Disk_2","Clusters +Z Disk2;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
    meClPosDisk1mz = theDMBE->book2D("position_siPixelClusters_mz_Disk_1","Clusters -Z Disk1;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
    meClPosDisk2mz = theDMBE->book2D("position_siPixelClusters_mz_Disk_2","Clusters -Z Disk2;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
    if (isUpgrade) {
      meClPosDisk3pz = theDMBE->book2D("position_siPixelClusters_pz_Disk_3","Clusters +Z Disk3;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
      meClPosDisk3mz = theDMBE->book2D("position_siPixelClusters_mz_Disk_3","Clusters -Z Disk3;Global X (cm);Global Y (cm)",80,-20.,20.,80,-20.,20.);
    }
    
    firstRun = false;
  }
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
  
  //if(modOn && !isUpgrade){
  if(!isUpgrade){
    MonitorElement* meReset = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_1");
    MonitorElement* meReset1 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_2");
    MonitorElement* meReset2 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_3");
    MonitorElement* meReset3 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_1");
    MonitorElement* meReset4 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_2");
    MonitorElement* meReset5 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_1");
    MonitorElement* meReset6 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_2");
    if(meReset && meReset->getEntries()>150000){
      meReset->Reset();
      meReset1->Reset();
      meReset2->Reset();
      meReset3->Reset();
      meReset4->Reset();
      meReset5->Reset();
      meReset6->Reset();
    }
  //}else if(modOn && isUpgrade){
  }else if(isUpgrade){
    MonitorElement* meReset = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_1");
    MonitorElement* meReset1 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_2");
    MonitorElement* meReset2 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_3");
    MonitorElement* meReset3 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_4");
    MonitorElement* meReset4 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_1");
    MonitorElement* meReset5 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_2");
    MonitorElement* meReset6 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_3");
    MonitorElement* meReset7 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_1");
    MonitorElement* meReset8 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_2");
    MonitorElement* meReset9 = theDMBE->get("Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_3");
    if(meReset && meReset->getEntries()>150000){
      meReset->Reset();
      meReset1->Reset();
      meReset2->Reset();
      meReset3->Reset();
      meReset4->Reset();
      meReset5->Reset();
      meReset6->Reset();
      meReset7->Reset();
      meReset8->Reset();
      meReset9->Reset();
    }
  }
  
  // get input data
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  input;
  iEvent.getByToken(srcToken_, input);

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry* tracker = &(* pDD);
//  const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detId) );

  //float iOrbitSec = iEvent.orbitNumber()/11223.;
  //int bx = iEvent.bunchCrossing();
  //long long tbx = (long long)iEvent.orbitNumber() * 3564 + bx;
  int lumiSection = (int)iEvent.luminosityBlock();
  int nEventFpixClusters = 0;


  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    int numberOfFpixClusters = (*struct_iter).second->fill(*input, tracker,  modOn, 
                                                           ladOn, layOn, phiOn, 
                                                           bladeOn, diskOn, ringOn, 
							   twoDimOn, reducedSet, smileyOn, isUpgrade);
    nEventFpixClusters = nEventFpixClusters + numberOfFpixClusters;    
    
  }

//  if(lumiSection>lumSec){ lumSec = lumiSection; nLumiSecs++; }
//  if(nEventFpixClusters>bigEventSize) nBigEvents++;
//  if(nLumiSecs%5==0){

  if(nEventFpixClusters>bigEventSize){
    MonitorElement* me = theDMBE->get("Pixel/bigFpixClusterEventRate");
    if(me){ 
      me->Fill(lumiSection,1./23.);    
    }
  }
  //std::cout<<"nEventFpixClusters: "<<nEventFpixClusters<<" , nLumiSecs: "<<nLumiSecs<<" , nBigEvents: "<<nBigEvents<<std::endl;
  
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

      
      if((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) ||
         (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap))){ 
        uint32_t id = detId();
        SiPixelClusterModule* theModule = new SiPixelClusterModule(id, ncols, nrows);
        if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
          if(isPIB) continue;
	  LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	  thePixelStructure.insert(pair<uint32_t,SiPixelClusterModule*> (id,theModule));
        }else if ( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (!isUpgrade) ) {
	  LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
          PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId(id)).halfCylinder();
          int disk   = PixelEndcapName(DetId(id)).diskName();
          int blade  = PixelEndcapName(DetId(id)).bladeName();
          int panel  = PixelEndcapName(DetId(id)).pannelName();
          int module = PixelEndcapName(DetId(id)).plaquetteName();
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
        } else if ( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (isUpgrade) ) {
          LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
          PixelEndcapNameUpgrade::HalfCylinder side = PixelEndcapNameUpgrade(DetId(id)).halfCylinder();
          int disk   = PixelEndcapNameUpgrade(DetId(id)).diskName();
          int blade  = PixelEndcapNameUpgrade(DetId(id)).bladeName();
          int panel  = PixelEndcapNameUpgrade(DetId(id)).pannelName();
          int module = PixelEndcapNameUpgrade(DetId(id)).plaquetteName();
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
        }//endif(Upgrade)
      }
    }
  }
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelClusterSource::bookMEs(){
  
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  theDMBE->setCurrentFolder("Pixel");
  char title[256]; snprintf(title, 256, "Rate of events with >%i FPIX clusters;LumiSection;Rate of large FPIX events per LS [Hz]",bigEventSize);
  bigFpixClusterEventRate = theDMBE->book1D("bigFpixClusterEventRate",title,5000,0.,5000.);


  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
    
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,0,isUpgrade)){
        (*struct_iter).second->book( conf_,0,twoDimOn,reducedSet,isUpgrade);
      } else {
        
        if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelClusterSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1,isUpgrade)){
	(*struct_iter).second->book( conf_,1,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if(layOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,2,isUpgrade)){
	(*struct_iter).second->book( conf_,2,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if(phiOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,3,isUpgrade)){
	(*struct_iter).second->book( conf_,3,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4,isUpgrade)){
	(*struct_iter).second->book( conf_,4,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,5,isUpgrade)){
	(*struct_iter).second->book( conf_,5,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,6,isUpgrade)){
	(*struct_iter).second->book( conf_,6,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
    if(smileyOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,7,isUpgrade)){
        (*struct_iter).second->book( conf_,7,twoDimOn,reducedSet,isUpgrade);
        } else {
        LogDebug ("PixelDQM") << "PROBLEM WITH BARREL-FOLDER\n";
      }
    }

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelClusterSource);
