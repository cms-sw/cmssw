// -*- C++ -*-
//
// Package:    SiPixelMonitorRawData
// Class:      SiPixelRawDataErrorSource
// 
/**\class 

 Description: 
 Produces histograms for error information generated at the raw2digi stage for the 
 pixel tracker.

 Implementation:
 Takes a DetSetVector<SiPixelRawDataError> as input, and uses it to populate  a folder 
 hierarchy (organized by detId) with histograms containing information about 
 the errors.  Uses SiPixelRawDataErrorModule class to book and fill individual folders.  
 Note that this source is different than other DQM sources in the creation of an 
 unphysical detId folder (detId=0xffffffff) to hold information about errors for which 
 there is no detId available (except the dummy detId given to it at raw2digi).

*/
//
// Original Author:  Andrew York
//
#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorSource.h"
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
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
//
#include <string>
#include <stdlib.h>

using namespace std;
using namespace edm;

SiPixelRawDataErrorSource::SiPixelRawDataErrorSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) ),
  saveFile( conf_.getUntrackedParameter<bool>("saveFile",false) ),
  isPIB( conf_.getUntrackedParameter<bool>("isPIB",false) ),
  slowDown( conf_.getUntrackedParameter<bool>("slowDown",false) ),
  reducedSet( conf_.getUntrackedParameter<bool>("reducedSet",false) ),
  modOn( conf_.getUntrackedParameter<bool>("modOn",true) ),
  ladOn( conf_.getUntrackedParameter<bool>("ladOn",false) ), 
  bladeOn( conf_.getUntrackedParameter<bool>("bladeOn",false) ),
  isUpgrade( conf_.getUntrackedParameter<bool>("isUpgrade",false) )
{
   theDMBE = edm::Service<DQMStore>().operator->();
   LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::SiPixelRawDataErrorSource: Got DQM BackEnd interface"<<endl;
}


SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource: Destructor"<<endl;
}


void SiPixelRawDataErrorSource::beginJob(){
  firstRun = true;
}

void SiPixelRawDataErrorSource::beginRun(const edm::Run& r, const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelRawDataErrorSource::beginRun - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Blade " << modOn << "/" << ladOn << "/" << bladeOn << std::endl;

  if(firstRun){
    eventNo = 0;
    // Build map
    buildStructure(iSetup);
    // Book Monitoring Elements
    bookMEs();
    
    firstRun = false;
  }
}


void SiPixelRawDataErrorSource::endJob(void){

  if(saveFile) {
    LogInfo ("PixelDQM") << " SiPixelRawDataErrorSource::endJob - Saving Root File " << std::endl;
    std::string outputFile = conf_.getParameter<std::string>("outputFile");
    theDMBE->save( outputFile.c_str() );
  }

}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  //std::cout<<"Event number: "<<eventNo<<std::endl;
  // get input data
  edm::Handle< edm::DetSetVector<SiPixelRawDataError> >  input;
  iEvent.getByLabel( src_, input );
  if (!input.isValid()) return; 

  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
   
  //float iOrbitSec = iEvent.orbitNumber()/11223.;
  //int bx = iEvent.bunchCrossing();
  //long long tbx = (long long)iEvent.orbitNumber() * 3564 + bx;
  int lumiSection = (int)iEvent.luminosityBlock();
  
  int nEventBPIXModuleErrors = 0; int nEventFPIXModuleErrors = 0; int nEventBPIXFEDErrors = 0; int nEventFPIXFEDErrors = 0;
  int nErrors = 0;

  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter2;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    int numberOfModuleErrors = (*struct_iter).second->fill(*input, modOn, ladOn, bladeOn);
    if(DetId( (*struct_iter).first ).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) nEventBPIXModuleErrors = nEventBPIXModuleErrors + numberOfModuleErrors;
    if(DetId( (*struct_iter).first ).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) nEventFPIXModuleErrors = nEventFPIXModuleErrors + numberOfModuleErrors;
    //cout<<"NErrors: "<<nEventBPIXModuleErrors<<" "<<nEventFPIXModuleErrors<<endl;
    nErrors = nErrors + numberOfModuleErrors;
    //if(nErrors>0) cout<<"MODULES: nErrors: "<<nErrors<<endl;
  }
  for (struct_iter2 = theFEDStructure.begin() ; struct_iter2 != theFEDStructure.end() ; struct_iter2++) {
    
    int numberOfFEDErrors = (*struct_iter2).second->fillFED(*input);
    if((*struct_iter2).first <= 31) nEventBPIXFEDErrors = nEventBPIXFEDErrors + numberOfFEDErrors; // (*struct_iter2).first >= 0, since (*struct_iter2).first is unsigned
    if((*struct_iter2).first >= 32 && (*struct_iter2).first <= 39) nEventFPIXFEDErrors = nEventFPIXFEDErrors + numberOfFEDErrors;    
    //cout<<"NFEDErrors: "<<nEventBPIXFEDErrors<<" "<<nEventFPIXFEDErrors<<endl;
    nErrors = nErrors + numberOfFEDErrors;
    //if(nErrors>0) cout<<"FEDS: nErrors: "<<nErrors<<endl;
  }
  MonitorElement* me = theDMBE->get("Pixel/AdditionalPixelErrors/byLumiErrors");
  if(me){
    me->setBinContent(0,eventNo);
    //cout<<"NErrors: "<<nEventBPIXModuleErrors<<" "<<nEventFPIXModuleErrors<<" "<<nEventBPIXFEDErrors<<" "<<nEventFPIXFEDErrors<<endl;
    //if(nEventBPIXModuleErrors+nEventBPIXFEDErrors>0) me->setBinContent(1,me->getBinContent(1)+nEventBPIXModuleErrors+nEventBPIXFEDErrors);
    //if(nEventFPIXModuleErrors+nEventFPIXFEDErrors>0) me->setBinContent(2,me->getBinContent(2)+nEventFPIXModuleErrors+nEventFPIXFEDErrors);
    if(nEventBPIXModuleErrors+nEventBPIXFEDErrors>0) me->Fill(0,1.);
    if(nEventFPIXModuleErrors+nEventFPIXFEDErrors>0) me->Fill(1,1.);
    //cout<<"histo: "<<me->getBinContent(0)<<" "<<me->getBinContent(1)<<" "<<me->getBinContent(2)<<endl;
  }  
  
  // Rate of errors per lumi section:
  MonitorElement* me1 = theDMBE->get("Pixel/AdditionalPixelErrors/errorRate");
  if(me1) me1->Fill(lumiSection, nErrors);

  // slow down...
  if(slowDown) usleep(100000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelRawDataErrorSource::buildStructure" ;
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;

  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){

    if( ((*it)->subDetector()==GeomDetEnumerators::PixelBarrel) || ((*it)->subDetector()==GeomDetEnumerators::PixelEndcap) ){
      DetId detId = (*it)->geographicalId();
      const GeomDetUnit      * geoUnit = pDD->idToDetUnit( detId );
      const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      int nrows = (pixDet->specificTopology()).nrows();
      int ncols = (pixDet->specificTopology()).ncolumns();

      if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        if(isPIB) continue;
        LogDebug ("PixelDQM") << " ---> Adding Barrel Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));

      }	else if( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (!isUpgrade)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	
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
		
	thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));
      }	else if( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (isUpgrade)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	
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
		
	thePixelStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));
      }//endif(isUpgrade)
    }
  }
  LogDebug ("PixelDQM") << " ---> Adding Module for Additional Errors " << endl;
  pair<int,int> fedIds (FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID);
  fedIds.first = 0;
  fedIds.second = 39;
  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    //std::cout<<"Adding FED module: "<<fedId<<std::endl;
    uint32_t id = static_cast<uint32_t> (fedId);
    SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id);
    theFEDStructure.insert(pair<uint32_t,SiPixelRawDataErrorModule*> (id,theModule));
  }
  
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::bookMEs(){
  //cout<<"Entering SiPixelRawDataErrorSource::bookMEs now: "<<endl;
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  theDMBE->setCurrentFolder("Pixel/AdditionalPixelErrors");
  char title[80]; sprintf(title, "By-LumiSection Error counters");
  byLumiErrors = theDMBE->book1D("byLumiErrors",title,2,0.,2.);
  byLumiErrors->setLumiFlag();
  char title1[80]; sprintf(title1, "Errors per LumiSection;LumiSection;NErrors");
  errorRate = theDMBE->book1D("errorRate",title1,5000,0.,5000.);
  
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter2;
  
  SiPixelFolderOrganizer theSiPixelFolder;
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    /// Create folder tree and book histograms 

    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,0,isUpgrade)) {
        (*struct_iter).second->book( conf_, 0 ,isUpgrade );
      }
      else {
        //std::cout<<"PIB! not booking histograms for non-PIB modules!"<<std::endl;
        if(!isPIB) throw cms::Exception("LogicError")
                       << "[SiPixelRawDataErrorSource::bookMEs] Creation of DQM folder failed";
      }
    }
    
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1,isUpgrade)) {
        (*struct_iter).second->book( conf_, 1 , isUpgrade );
      }
      else {
        LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4,isUpgrade)) {
        (*struct_iter).second->book( conf_, 4 ,isUpgrade );
      }
      else {
        LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    
  }//for loop

  for(struct_iter2 = theFEDStructure.begin(); struct_iter2 != theFEDStructure.end(); struct_iter2++){
    /// Create folder tree for errors without detId and book histograms 
    if(theSiPixelFolder.setFedFolder((*struct_iter2).first)) {
      (*struct_iter2).second->bookFED( conf_ );
    }
    else {
      throw cms::Exception("LogicError")
	<< "[SiPixelRawDataErrorSource::bookMEs] Creation of DQM folder failed";
    }

  }
  //cout<<"...leaving SiPixelRawDataErrorSource::bookMEs now! "<<endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelRawDataErrorSource);
