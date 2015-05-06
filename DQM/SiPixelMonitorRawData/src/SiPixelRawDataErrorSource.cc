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
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
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
  src_( consumes<DetSetVector<SiPixelRawDataError> >( conf_.getParameter<edm::InputTag>( "src" ) ) ),
  saveFile( conf_.getUntrackedParameter<bool>("saveFile",false) ),
  isPIB( conf_.getUntrackedParameter<bool>("isPIB",false) ),
  slowDown( conf_.getUntrackedParameter<bool>("slowDown",false) ),
  reducedSet( conf_.getUntrackedParameter<bool>("reducedSet",false) ),
  modOn( conf_.getUntrackedParameter<bool>("modOn",true) ),
  ladOn( conf_.getUntrackedParameter<bool>("ladOn",false) ), 
  bladeOn( conf_.getUntrackedParameter<bool>("bladeOn",false) ),
  isUpgrade( conf_.getUntrackedParameter<bool>("isUpgrade",false) )
{
  firstRun = true;
  LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::SiPixelRawDataErrorSource: Got DQM BackEnd interface"<<endl;
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  inputSourceToken_ = consumes<FEDRawDataCollection>(conf_.getUntrackedParameter<string>("inputSource", "source"));
}


SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelRawDataErrorSource::~SiPixelRawDataErrorSource: Destructor"<<endl;
}

void SiPixelRawDataErrorSource::dqmBeginRun(const edm::Run& r, const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") << " SiPixelRawDataErrorSource::beginRun - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Blade " << modOn << "/" << ladOn << "/" << bladeOn << std::endl;

  if(firstRun){
    eventNo = 0;
    
    firstRun = false;
  }

  // Build map
  buildStructure(iSetup);
}

void SiPixelRawDataErrorSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & iSetup){
  // Book Monitoring Elements
  bookMEs(iBooker);
}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  //check feds in readout
  if(eventNo==1){
    // check if any Pixel FED is in readout:
    edm::Handle<FEDRawDataCollection> rawDataHandle;
    iEvent.getByToken(inputSourceToken_, rawDataHandle);
    if(!rawDataHandle.isValid()){
      edm::LogInfo("SiPixelRawDataErrorSource") << "inputsource is empty";
    }
    else{
      const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
      for(int i = 0; i != 40; i++){
        if(rawDataCollection.FEDData(i).size() && rawDataCollection.FEDData(i).data()) fedcounter->setBinContent(i+1,1);
      }
    }
  }
  // get input data
  edm::Handle< DetSetVector<SiPixelRawDataError> >  input;
  iEvent.getByToken( src_, input );
  if (!input.isValid()) return; 

  int lumiSection = (int)iEvent.luminosityBlock();
  
  int nEventBPIXModuleErrors = 0; int nEventFPIXModuleErrors = 0; int nEventBPIXFEDErrors = 0; int nEventFPIXFEDErrors = 0;
  int nErrors = 0;

  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter2;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    
    int numberOfModuleErrors = (*struct_iter).second->fill(*input, &meMapFEDs_, modOn, ladOn, bladeOn);
    if(DetId( (*struct_iter).first ).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) nEventBPIXModuleErrors = nEventBPIXModuleErrors + numberOfModuleErrors;
    if(DetId( (*struct_iter).first ).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) nEventFPIXModuleErrors = nEventFPIXModuleErrors + numberOfModuleErrors;
    //cout<<"NErrors: "<<nEventBPIXModuleErrors<<" "<<nEventFPIXModuleErrors<<endl;
    nErrors = nErrors + numberOfModuleErrors;
    //if(nErrors>0) cout<<"MODULES: nErrors: "<<nErrors<<endl;
  }
  for (struct_iter2 = theFEDStructure.begin() ; struct_iter2 != theFEDStructure.end() ; struct_iter2++) {
    
    int numberOfFEDErrors = (*struct_iter2).second->fillFED(*input, &meMapFEDs_);
    if((*struct_iter2).first <= 31) nEventBPIXFEDErrors = nEventBPIXFEDErrors + numberOfFEDErrors; // (*struct_iter2).first >= 0, since (*struct_iter2).first is unsigned
    if((*struct_iter2).first >= 32 && (*struct_iter2).first <= 39) nEventFPIXFEDErrors = nEventFPIXFEDErrors + numberOfFEDErrors;    
    //cout<<"NFEDErrors: "<<nEventBPIXFEDErrors<<" "<<nEventFPIXFEDErrors<<endl;
    nErrors = nErrors + numberOfFEDErrors;
    //if(nErrors>0) cout<<"FEDS: nErrors: "<<nErrors<<endl;
  }
  if(byLumiErrors){
    byLumiErrors->setBinContent(0,eventNo);
    //cout<<"NErrors: "<<nEventBPIXModuleErrors<<" "<<nEventFPIXModuleErrors<<" "<<nEventBPIXFEDErrors<<" "<<nEventFPIXFEDErrors<<endl;
    if(nEventBPIXModuleErrors+nEventBPIXFEDErrors>0) byLumiErrors->Fill(0,1.);
    if(nEventFPIXModuleErrors+nEventFPIXFEDErrors>0) byLumiErrors->Fill(1,1.);
    //cout<<"histo: "<<byLumiErrors->getBinContent(0)<<" "<<byLumiErrors->getBinContent(1)<<" "<<byLumiErrors->getBinContent(2)<<endl;
  }  
  
  // Rate of errors per lumi section:
  if(errorRate) errorRate->Fill(lumiSection, nErrors);

  // slow down...
  if(slowDown) usleep(100000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelRawDataErrorSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelRawDataErrorSource::buildStructure" ;


  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<TrackerTopology> tTopoHandle;

  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);

  const TrackerTopology *pTT = tTopoHandle.product();


  LogVerbatim ("PixelDQM") << " *** Geometry node for TrackerGeom is  "<<&(*pDD)<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->dets().size() <<" detectors"<<std::endl;
  LogVerbatim ("PixelDQM") << " *** I have " << pDD->detTypes().size() <<" types"<<std::endl;

  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){

    if( GeomDetEnumerators::isTrackerPixel((*it)->subDetector())) {

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

      }	else if( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) ) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelRawDataErrorModule* theModule = new SiPixelRawDataErrorModule(id, ncols, nrows);
	
        PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId(id), pTT, isUpgrade).halfCylinder();
        int disk   = PixelEndcapName(DetId(id), pTT, isUpgrade).diskName();
        int blade  = PixelEndcapName(DetId(id), pTT, isUpgrade).bladeName();
        int panel  = PixelEndcapName(DetId(id), pTT, isUpgrade).pannelName();
        int module = PixelEndcapName(DetId(id), pTT, isUpgrade).plaquetteName();

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

      }	
    }//MAIN_IF
  }//FOR_LOOP

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
void SiPixelRawDataErrorSource::bookMEs(DQMStore::IBooker & iBooker){
  iBooker.setCurrentFolder(topFolderName_+"/EventInfo/DAQContents");
  char title0[80]; sprintf(title0, "FED isPresent;FED ID;isPresent");
  fedcounter = iBooker.book1D("fedcounter",title0,40,-0.5,39.5);
  iBooker.setCurrentFolder(topFolderName_+"/AdditionalPixelErrors");
  char title[80]; sprintf(title, "By-LumiSection Error counters");
  byLumiErrors = iBooker.book1D("byLumiErrors",title,2,0.,2.);
  byLumiErrors->setLumiFlag();
  char title1[80]; sprintf(title1, "Errors per LumiSection;LumiSection;NErrors");
  errorRate = iBooker.book1D("errorRate",title1,5000,0.,5000.);
  
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter;
  std::map<uint32_t,SiPixelRawDataErrorModule*>::iterator struct_iter2;
  
  SiPixelFolderOrganizer theSiPixelFolder(false);
  
  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    /// Create folder tree and book histograms 

    if(modOn){
      if(!theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,0,isUpgrade)) {
        //std::cout<<"PIB! not booking histograms for non-PIB modules!"<<std::endl;
        if(!isPIB) throw cms::Exception("LogicError")
                       << "[SiPixelRawDataErrorSource::bookMEs] Creation of DQM folder failed";
      }
    }
    
    if(ladOn){
      if(!theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,1,isUpgrade)) {
        LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
    }
    
    if(bladeOn){
      if(!theSiPixelFolder.setModuleFolder(iBooker,(*struct_iter).first,4,isUpgrade)) {
        LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    
  }//for loop

  for(struct_iter2 = theFEDStructure.begin(); struct_iter2 != theFEDStructure.end(); struct_iter2++){
    /// Create folder tree for errors without detId and book histograms 
    if(!theSiPixelFolder.setFedFolder(iBooker,(*struct_iter2).first)) {
      throw cms::Exception("LogicError")
	<< "[SiPixelRawDataErrorSource::bookMEs] Creation of DQM folder failed";
    }

  }

  //Booking FED histograms
  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = conf_.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );

  for (uint32_t id = 0; id < 40; id++){
    char temp [50];
    sprintf( temp, (topFolderName_+"/AdditionalPixelErrors/FED_%d").c_str(),id);
    iBooker.cd(temp);
    // Types of errors
    hid = theHistogramId->setHistoId("errorType",id);
    meErrorType_[id] = iBooker.book1D(hid,"Type of errors",15,24.5,39.5);
    meErrorType_[id]->setAxisTitle("Type of errors",1);
    // Number of errors
    hid = theHistogramId->setHistoId("NErrors",id);
    meNErrors_[id] = iBooker.book1D(hid,"Number of errors",36,0.,36.);
    meNErrors_[id]->setAxisTitle("Number of errors",1);
    // Type of FIFO full (errorType = 28).  FIFO 1 is 1-5 (where fullType = channel of FIFO 1), 
    // fullType = 6 signifies FIFO 2 nearly full, 7 signifies trigger FIFO nearly full, 8 
    // indicates an unexpected result
    hid = theHistogramId->setHistoId("fullType",id);
    meFullType_[id] = iBooker.book1D(hid,"Type of FIFO full",7,0.5,7.5);
    meFullType_[id]->setAxisTitle("FIFO type",1);
    // For error type 30, the type of problem encoded in the TBM trailer
    // 0 = stack full, 1 = Pre-cal issued, 2 = clear trigger counter, 3 = sync trigger, 
    // 4 = sync trigger error, 5 = reset ROC, 6 = reset TBM, 7 = no token bit pass
    hid = theHistogramId->setHistoId("TBMMessage",id);
    meTBMMessage_[id] = iBooker.book1D(hid,"TBM trailer message",8,-0.5,7.5);
    meTBMMessage_[id]->setAxisTitle("TBM message",1);
    // For error type 30, the type of problem encoded in the TBM error trailer 0 = none
    // 1 = data stream too long, 2 = FSM errors, 3 = invalid # of ROCs, 4 = multiple
    hid = theHistogramId->setHistoId("TBMType",id);
    meTBMType_[id] = iBooker.book1D(hid,"Type of TBM trailer",5,-0.5,4.5);
    meTBMType_[id]->setAxisTitle("TBM Type",1);
    // For error type 31, the event number of the TBM header with the error
    hid = theHistogramId->setHistoId("EvtNbr",id);
    meEvtNbr_[id] = iBooker.book1D(hid,"Event number",1,0,1);
    // For errorType = 34, datastream size according to error word
    hid = theHistogramId->setHistoId("evtSize",id);
    meEvtSize_[id] = iBooker.book1D(hid,"Event size",1,0,1);
    //
    hid = theHistogramId->setHistoId("FedChNErr",id);
    meFedChNErr_[id] = iBooker.book1D(hid,"Number of errors per FED channel",37,0,37);
    meFedChNErr_[id]->setAxisTitle("FED channel",1);
    //
    hid = theHistogramId->setHistoId("FedChLErr",id);
    meFedChLErr_[id] = iBooker.book1D(hid,"Last error per FED channel",37,0,37);
    meFedChLErr_[id]->setAxisTitle("FED channel",1);
    //
    hid = theHistogramId->setHistoId("FedETypeNErr", id);
    meFedETypeNErr_[id] = iBooker.book1D(hid,"Number of errors per type",21,0,21);
    meFedETypeNErr_[id]->setAxisTitle("Error type",1);
  }
  //Add the booked histograms to the histogram map for booking
  meMapFEDs_["meErrorType_"] = meErrorType_;
  meMapFEDs_["meNErrors_"] = meNErrors_;
  meMapFEDs_["meFullType_"] = meFullType_;
  meMapFEDs_["meTBMMessage_"] = meTBMMessage_;
  meMapFEDs_["meTBMType_"] = meTBMType_;
  meMapFEDs_["meEvtNbr_"] = meEvtNbr_;
  meMapFEDs_["meEvtSize_"] = meEvtSize_;
  meMapFEDs_["meFedChNErr_"] = meFedChNErr_;
  meMapFEDs_["meFedChLErr_"] = meFedChLErr_;
  meMapFEDs_["meFedETypeNErr_"] = meFedETypeNErr_;

  //cout<<"...leaving SiPixelRawDataErrorSource::bookMEs now! "<<endl;
}

DEFINE_FWK_MODULE(SiPixelRawDataErrorSource);
