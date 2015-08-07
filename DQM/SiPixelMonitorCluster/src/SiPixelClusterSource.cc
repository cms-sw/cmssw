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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
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
  isUpgrade( conf_.getUntrackedParameter<bool>("isUpgrade",false) ),
  noOfLayers(0),
  noOfDisks(0)
{
   LogInfo ("PixelDQM") << "SiPixelClusterSource::SiPixelClusterSource: Got DQM BackEnd interface"<<endl;

   //set Token(-s)
   srcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(conf_.getParameter<edm::InputTag>("src"));
   firstRun = true;
   topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
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



void SiPixelClusterSource::dqmBeginRun(const edm::Run& r, const edm::EventSetup& iSetup){
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
 
    firstRun = false;
  }
}

void SiPixelClusterSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, const edm::EventSetup& iSetup){
  bookMEs(iBooker, iSetup);
  // Book occupancy maps in global coordinates for all clusters:
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OffTrack");
  //bpix
  std::stringstream ss1, ss2;
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "position_siPixelClusters_Layer_" << i;
    ss2.str(std::string()); ss2 << "Clusters Layer" << i << ";Global Z (cm);Global #phi";
    meClPosLayer.push_back(iBooker.book2D(ss1.str(),ss2.str(),200,-30.,30.,128,-3.2,3.2));
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "position_siPixelClusters_pz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters +Z Disk" << i << ";Global X (cm);Global Y (cm)";
    meClPosDiskpz.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
    ss1.str(std::string()); ss1 << "position_siPixelClusters_mz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters -Z Disk" << i << ";Global X (cm);Global Y (cm)";
    meClPosDiskmz.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
  }
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelClusterSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  
  if(meClPosLayer.at(0) && meClPosLayer.at(0)->getEntries()>150000){
    for (int i = 0; i < noOfLayers; i++)
    {
      meClPosLayer.at(i)->Reset();
    }
    for (int i = 0; i < noOfDisks; i++)
    {
      meClPosDiskpz.at(i)->Reset();
      meClPosDiskmz.at(i)->Reset();
    }
  }
  
  // get input data
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  input;
  iEvent.getByToken(srcToken_, input);

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry* tracker = &(* pDD);

  int lumiSection = (int)iEvent.luminosityBlock();
  int nEventFpixClusters = 0;


  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    int numberOfFpixClusters = (*struct_iter).second->fill(*input, tracker,  
							   meClPosLayer,
							   meClPosDiskpz,
							   meClPosDiskmz,
							   modOn, ladOn, layOn, phiOn, 
                                                           bladeOn, diskOn, ringOn, 
							   twoDimOn, reducedSet, smileyOn, isUpgrade);
    nEventFpixClusters = nEventFpixClusters + numberOfFpixClusters;    
    
  }

  if(nEventFpixClusters>bigEventSize){
    if (bigFpixClusterEventRate){
      bigFpixClusterEventRate->Fill(lumiSection,1./23.);
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

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();

  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;
  
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    
    if(dynamic_cast<PixelGeomDetUnit const *>((*it))!=0){

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
          int layer = PixelBarrelName(DetId(id),pTT,isUpgrade).layerName();
          if (layer > noOfLayers) noOfLayers = layer;
	  thePixelStructure.insert(pair<uint32_t,SiPixelClusterModule*> (id,theModule));
        }else if ( detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap ) ) {
	  LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
          PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId(id),pTT,isUpgrade).halfCylinder();
          int disk   = PixelEndcapName(DetId(id),pTT,isUpgrade).diskName();
	  if (disk>noOfDisks) noOfDisks=disk;
          int blade  = PixelEndcapName(DetId(id),pTT,isUpgrade).bladeName();
          int panel  = PixelEndcapName(DetId(id),pTT,isUpgrade).pannelName();
          int module = PixelEndcapName(DetId(id),pTT,isUpgrade).plaquetteName();
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
  }
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelClusterSource::bookMEs(DQMStore::IBooker & iBooker, const edm::EventSetup& iSetup){
  
  // Get DQM interface
  iBooker.setCurrentFolder(topFolderName_);
  char title[256]; snprintf(title, 256, "Rate of events with >%i FPIX clusters;LumiSection;Rate of large FPIX events per LS [Hz]",bigEventSize);
  bigFpixClusterEventRate = iBooker.book1D("bigFpixClusterEventRate",title,5000,0.,5000.);


  std::map<uint32_t,SiPixelClusterModule*>::iterator struct_iter;
    
  SiPixelFolderOrganizer theSiPixelFolder(false);
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,0,isUpgrade)){
        (*struct_iter).second->book( conf_,iSetup,iBooker,0,twoDimOn,reducedSet,isUpgrade);
      } else {
        
        if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelClusterSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,1,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,1,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    if(layOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,2,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,2,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }
    if(phiOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,3,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,3,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,4,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,4,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,5,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,5,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,6,isUpgrade)){
	(*struct_iter).second->book( conf_,iSetup,iBooker,6,twoDimOn,reducedSet,isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
    if(smileyOn){
      if(theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,7,isUpgrade)){
        (*struct_iter).second->book( conf_,iSetup,iBooker,7,twoDimOn,reducedSet,isUpgrade);
        } else {
        LogDebug ("PixelDQM") << "PROBLEM WITH BARREL-FOLDER\n";
      }
    }

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelClusterSource);
