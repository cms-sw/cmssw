// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelDigiSource
// 
/**\class 

 Description: Pixel DQM source for Digis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiSource.cc,v 1.45 2010/08/06 14:44:17 duggan Exp $
//
//
#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiSource.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
#include <iostream>
#include <fstream>

using namespace std;
using namespace edm;

SiPixelDigiSource::SiPixelDigiSource(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  src_( conf_.getParameter<edm::InputTag>( "src" ) ),
  saveFile( conf_.getUntrackedParameter<bool>("saveFile",false) ),
  isPIB( conf_.getUntrackedParameter<bool>("isPIB",false) ),
  slowDown( conf_.getUntrackedParameter<bool>("slowDown",false) ),
  modOn( conf_.getUntrackedParameter<bool>("modOn",true) ),
  twoDimOn( conf_.getUntrackedParameter<bool>("twoDimOn",true) ),
  twoDimModOn( conf_.getUntrackedParameter<bool>("twoDimModOn",true) ),
  twoDimOnlyLayDisk( conf_.getUntrackedParameter<bool>("twoDimOnlyLayDisk",false) ),
  hiRes( conf_.getUntrackedParameter<bool>("hiRes",false) ),
  reducedSet( conf_.getUntrackedParameter<bool>("reducedSet",false) ),
  ladOn( conf_.getUntrackedParameter<bool>("ladOn",false) ), 
  layOn( conf_.getUntrackedParameter<bool>("layOn",false) ), 
  phiOn( conf_.getUntrackedParameter<bool>("phiOn",false) ), 
  ringOn( conf_.getUntrackedParameter<bool>("ringOn",false) ), 
  bladeOn( conf_.getUntrackedParameter<bool>("bladeOn",false) ), 
  diskOn( conf_.getUntrackedParameter<bool>("diskOn",false) ),
  bigEventSize( conf_.getUntrackedParameter<int>("bigEventSize",1000) )
{
   theDMBE = edm::Service<DQMStore>().operator->();
   LogInfo ("PixelDQM") << "SiPixelDigiSource::SiPixelDigiSource: Got DQM BackEnd interface"<<endl;
}


SiPixelDigiSource::~SiPixelDigiSource()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelDigiSource::~SiPixelDigiSource: Destructor"<<endl;
}


void SiPixelDigiSource::beginJob(){
  firstRun = true;  
  // find a FED# for the current detId:
  ifstream infile(edm::FileInPath("DQM/SiPixelMonitorClient/test/detId.dat").fullPath().c_str(),ios::in);
  int nModsInFile=0;
  assert(!infile.fail());
  while(!infile.eof()&&nModsInFile<1440) {
    infile >> I_name[nModsInFile] >> I_detId[nModsInFile] >> I_fedId[nModsInFile] >> I_linkId[nModsInFile] ;
    //cout<<nModsInFile<<" , "<<I_name[nModsInFile]<<" , "<<I_detId[nModsInFile]<<" , "<<I_fedId[nModsInFile]<<" , "<<I_linkId[nModsInFile]<<endl; ;
    nModsInFile++;
  }
  infile.close();
}

void SiPixelDigiSource::beginRun(const edm::Run& r, const edm::EventSetup& iSetup){
  LogInfo ("PixelDQM") << " SiPixelDigiSource::beginJob - Initialisation ... " << std::endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
		       << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
		       << ringOn << std::endl;
  
  LogInfo ("PixelDQM") << "2DIM IS " << twoDimOn << " and set to high resolution? " << hiRes << "\n";

  if(firstRun){
    eventNo = 0;
    lumSec = 0;
    nLumiSecs = 0;
    nBigEvents = 0;
    nBPIXDigis = 0; 
    nFPIXDigis = 0;
    for(int i=0; i!=40; i++) nDigisPerFed[i]=0;  
    
    // Build map
    buildStructure(iSetup);
    // Book Monitoring Elements
    bookMEs();
    firstRun = false;
  }
}


void SiPixelDigiSource::endJob(void){

  if(saveFile) {
    LogInfo ("PixelDQM") << " SiPixelDigiSource::endJob - Saving Root File " << std::endl;
    std::string outputFile = conf_.getParameter<std::string>("outputFile");
    theDMBE->save( outputFile.c_str() );
  }

}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelDigiSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  eventNo++;
  //cout<<"BIGFATEVENTNUMBER: "<<eventNo<<endl;

  // get input data
  edm::Handle< edm::DetSetVector<PixelDigi> >  input;
  iEvent.getByLabel( src_, input );
  if (!input.isValid()) return; 
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  
  //float iOrbitSec = iEvent.orbitNumber()/11223.;
  int bx = iEvent.bunchCrossing();
  //long long tbx = (long long)iEvent.orbitNumber() * 3564 + bx;
  int lumiSection = (int)iEvent.luminosityBlock();
  int nEventDigis = 0; int nActiveModules = 0;
  //int nEventBPIXDigis = 0; int nEventFPIXDigis = 0;
  
  if(modOn){
    MonitorElement* meReset = theDMBE->get("Pixel/averageDigiOccupancy");
    //if(meReset && eventNo%1000==0){
    if(meReset && lumiSection%8==0){
      meReset->Reset();
      nBPIXDigis = 0; 
      nFPIXDigis = 0;
      for(int i=0; i!=40; i++) nDigisPerFed[i]=0;  
    }
  }
  
  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
  for(int i=0; i!=192; i++) numberOfDigis[i]=0;
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    int numberOfDigisMod = (*struct_iter).second->fill(*input, modOn, 
				ladOn, layOn, phiOn, 
				bladeOn, diskOn, ringOn, 
				twoDimOn, reducedSet, twoDimModOn, twoDimOnlyLayDisk);
    if(numberOfDigisMod>0){
      nEventDigis = nEventDigis + numberOfDigisMod;  
      nActiveModules++;  
      bool barrel = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
      bool endcap = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
      //if((*struct_iter).first >= 302055684 && (*struct_iter).first <= 302197792 ){ // Barrel
      if(barrel){ // Barrel
        //cout<<"AAbpix: "<<numberOfDigisMod<<" + "<<nBPIXDigis<<" = ";
        nBPIXDigis = nBPIXDigis + numberOfDigisMod;
	//cout<<nBPIXDigis<<endl;
        for(int i=0; i!=768; i++){
          //cout<<"\t\t\t bpix: "<<i<<" , "<<(*struct_iter).first<<" , "<<I_detId[i]<<endl;
          if((*struct_iter).first == I_detId[i]){
	    //if(I_fedId[i]>=32&&I_fedId[i]<=39) std::cout<<"Attention: a BPIX module matched to an FPIX FED!"<<std::endl;
	    nDigisPerFed[I_fedId[i]]=nDigisPerFed[I_fedId[i]]+numberOfDigisMod;
	    //cout<<"BPIX: "<<i<<" , "<<I_fedId[i]<<" , "<<numberOfDigisMod<<" , "<<nDigisPerFed[I_fedId[i]]<<endl;
	    i=767;
	  }
        }
      //}else if((*struct_iter).first >= 343999748 && (*struct_iter).first <= 352477708 ){ // Endcap
      }else if(endcap){ // Endcap
        //cout<<"AAfpix: "<<nFPIXDigis<<" = ";
        nFPIXDigis = nFPIXDigis + numberOfDigisMod;
	//cout<<nFPIXDigis<<endl;
        PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId((*struct_iter).first)).halfCylinder();
	int disk = PixelEndcapName(DetId((*struct_iter).first)).diskName();
	int blade = PixelEndcapName(DetId((*struct_iter).first)).bladeName();
        int panel = PixelEndcapName(DetId((*struct_iter).first)).pannelName();
	//std::cout<<"Endcap: "<<side<<" , "<<disk<<" , "<<blade<<" , "<<panel<<" , "<<std::endl;
	int iter=-1; int i=0;
	if(side==PixelEndcapName::mI){
	  if(disk==1){
	    i=0;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }else if(disk==2){
	    i=24;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }
	}else if(side==PixelEndcapName::mO){
	  if(disk==1){
	    i=48;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }else if(disk==2){
	    i=72;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }
	}else if(side==PixelEndcapName::pI){
	  if(disk==1){
	    i=96;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }else if(disk==2){
	    i=120;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }
	}else if(side==PixelEndcapName::pO){
	  if(disk==1){
	    i=145;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }else if(disk==2){
	    i=169;
	    if(blade==1){ if(panel==1) iter=i; else if(panel==2) iter=i+1; }
	    if(blade==2){ if(panel==1) iter=i+2; else if(panel==2) iter=i+3; }
	    if(blade==3){ if(panel==1) iter=i+4; else if(panel==2) iter=i+5; }
	    if(blade==4){ if(panel==1) iter=i+6; else if(panel==2) iter=i+7; }
	    if(blade==5){ if(panel==1) iter=i+8; else if(panel==2) iter=i+9; }
	    if(blade==6){ if(panel==1) iter=i+10; else if(panel==2) iter=i+11; }
	    if(blade==7){ if(panel==1) iter=i+12; else if(panel==2) iter=i+13; }
	    if(blade==8){ if(panel==1) iter=i+14; else if(panel==2) iter=i+15; }
	    if(blade==9){ if(panel==1) iter=i+16; else if(panel==2) iter=i+17; }
	    if(blade==10){ if(panel==1) iter=i+18; else if(panel==2) iter=i+19; }
	    if(blade==11){ if(panel==1) iter=i+20; else if(panel==2) iter=i+21; }
	    if(blade==12){ if(panel==1) iter=i+22; else if(panel==2) iter=i+23; }
	  }
	}
	//std::cout<<"status: "<<iter<<","<<side<<","<<disk<<","<<blade<<","<<panel<<std::endl;       
	numberOfDigis[iter]=numberOfDigis[iter]+numberOfDigisMod;
        for(int i=768; i!=1440; i++){
          //cout<<"\t\t\t fpix: "<<i<<" , "<<(*struct_iter).first<<" , "<<I_detId[i]<<endl;
          if((*struct_iter).first == I_detId[i]){
	    //if(I_fedId[i]<32||I_fedId[i]>39) std::cout<<"Attention: an FPIX module matched to a BPIX FED!"<<std::endl;
	    nDigisPerFed[I_fedId[i]]=nDigisPerFed[I_fedId[i]]+numberOfDigisMod;
	    //cout<<"FPIX: "<<i<<" , "<<I_fedId[i]<<" , "<<nDigisPerFed[I_fedId[i]]<<endl;
	    i=1439;
	  }
        }
      } //endif Barrel/Endcap
      //cout<<"numberOfDigis: "<<numberOfDigisMod<<" , nBPIXDigis: "<<nBPIXDigis<<" , nFPIXDigis: "<<nFPIXDigis<<endl;
      // digi occupancy per individual FED channel:
    } // endif any digis in this module
  } // endfor loop over all modules
  
//  if(lumiSection>lumSec){ lumSec = lumiSection; nLumiSecs++; }
//  if(nEventDigis>bigEventSize) nBigEvents++;
//  if(nLumiSecs%5==0){
  MonitorElement* me; MonitorElement* me1;
  
  me=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_Endcap");
  if(me){ for(int j=0; j!=192; j++) if(numberOfDigis[j]>0) me->Fill((float)numberOfDigis[j]);}
  
  // Rate of events with >N digis:
  if(nEventDigis>bigEventSize){
    me = theDMBE->get("Pixel/bigEventRate");
    if(me) me->Fill(lumiSection,1./23.);    
  }
  //std::cout<<"nEventDigis: "<<nEventDigis<<" , nLumiSecs: "<<nLumiSecs<<" , nBigEvents: "<<nBigEvents<<std::endl;
  
  // Rate of pixel events and total number of pixel events per BX:
  if(nActiveModules>=4){
    me = theDMBE->get("Pixel/pixEvtsPerBX");
    if(me) me->Fill(float(bx));
    me1 = theDMBE->get("Pixel/pixEventRate");
    if(me1) me1->Fill(lumiSection, 1./23.);
  }
  
  // Actual digi occupancy in a FED compared to average digi occupancy per FED
  me = theDMBE->get("Pixel/averageDigiOccupancy");
  me1 = theDMBE->get("Pixel/avgfedDigiOccvsLumi");
  if(me){
    int maxfed=0;
    for(int i=0; i!=32; i++){
      if(nDigisPerFed[i]>maxfed) maxfed=nDigisPerFed[i];
    }
    for(int i=0; i!=40; i++){
      float averageOcc = 0.;
      if(i<32){
        float averageBPIXFed = float(nBPIXDigis-maxfed)/31.;
	if(averageBPIXFed>0.) averageOcc = nDigisPerFed[i]/averageBPIXFed;
	//cout<<"\t BPIX i: "<<i<<" , "<<nBPIXDigis<<" , "<<averageBPIXFed<<" , "<<nDigisPerFed[i]<<" , "<<averageOcc<<endl;
      }else{
        float averageFPIXFed = float(nFPIXDigis)/8.;
	if(averageFPIXFed>0.) averageOcc = nDigisPerFed[i]/averageFPIXFed;
	//cout<<"\t FPIX i: "<<i<<" , "<<nFPIXDigis<<" , "<<averageFPIXFed<<" , "<<nDigisPerFed[i]<<" , "<<averageOcc<<endl;
      }
      me->setBinContent(i+1,averageOcc);
      int lumiSections8 = int(lumiSection/8);
      if (modOn){
	if (me1){
	  me1->setBinContent(1+lumiSections8, i+1, averageOcc);
	}//endif me1
      }//endif modOn
    }
  }
  
  // slow down...
  if(slowDown) usleep(10000);
  
}

//------------------------------------------------------------------
// Build data structure
//------------------------------------------------------------------
void SiPixelDigiSource::buildStructure(const edm::EventSetup& iSetup){

  LogInfo ("PixelDQM") <<" SiPixelDigiSource::buildStructure" ;
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
	SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);
	thePixelStructure.insert(pair<uint32_t,SiPixelDigiModule*> (id,theModule));

      }	else if(detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);
       
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
	
	thePixelStructure.insert(pair<uint32_t,SiPixelDigiModule*> (id,theModule));
      }

    }
  }
  LogInfo ("PixelDQM") << " *** Pixel Structure Size " << thePixelStructure.size() << endl;
}
//------------------------------------------------------------------
// Book MEs
//------------------------------------------------------------------
void SiPixelDigiSource::bookMEs(){
  
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  theDMBE->setCurrentFolder("Pixel");
  char title[80]; sprintf(title, "Rate of events with >%i digis;LumiSection;Rate [Hz]",bigEventSize);
  bigEventRate = theDMBE->book1D("bigEventRate",title,5000,0.,5000.);
  char title1[80]; sprintf(title1, "Pixel events vs. BX;BX;# events");
  pixEvtsPerBX = theDMBE->book1D("pixEvtsPerBX",title1,3565,0.,3565.);
  char title2[80]; sprintf(title2, "Rate of Pixel events;LumiSection;Rate [Hz]");
  pixEventRate = theDMBE->book1D("pixEventRate",title2,5000,0.,5000.);
  char title3[80]; sprintf(title3, "Average digi occupancy per FED;FED;NDigis/<NDigis>");
  averageDigiOccupancy = theDMBE->book1D("averageDigiOccupancy",title3,40,-0.5,39.5);
  if(modOn){
    char title4[80]; sprintf(title4, "FED Digi Occupancy (NDigis/<NDigis>) vs LumiSections;Lumi Section;FED");
    avgfedDigiOccvsLumi = theDMBE->book2D ("avgfedDigiOccvsLumi", title4, 400,0., 3200., 40, -0.5, 39.5);
  }  
  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
 
  SiPixelFolderOrganizer theSiPixelFolder;

  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first)){
	(*struct_iter).second->book( conf_,0,twoDimOn,hiRes, reducedSet, twoDimModOn);
      } else {

	if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1)){
	(*struct_iter).second->book( conf_,1,twoDimOn,hiRes, reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
   
    }
    if(layOn || twoDimOnlyLayDisk){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,2)){
	(*struct_iter).second->book( conf_,2,twoDimOn,hiRes, reducedSet, twoDimOnlyLayDisk);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }

    if(phiOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,3)){
	(*struct_iter).second->book( conf_,3,twoDimOn,hiRes, reducedSet);
	} else {
        LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4)){
	(*struct_iter).second->book( conf_,4,twoDimOn,hiRes, reducedSet);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn || twoDimOnlyLayDisk){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,5)){
	(*struct_iter).second->book( conf_,5,twoDimOn,hiRes, reducedSet, twoDimOnlyLayDisk);
      } else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,6)){
	(*struct_iter).second->book( conf_,6,twoDimOn,hiRes, reducedSet);
      } else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
  }
  std::string currDir = theDMBE->pwd();
  theDMBE->cd("Pixel/Barrel");
  meNDigisCOMBBarrel_ = theDMBE->book1D("ALLMODS_ndigisCOMB_Barrel","Number of Digis",200,0.,400.);
  meNDigisCOMBBarrel_->setAxisTitle("Number of digis per module per event",1);
  meNDigisCHANBarrel_ = theDMBE->book1D("ALLMODS_ndigisCHAN_Barrel","Number of Digis",200,0.,400.);
  meNDigisCHANBarrel_->setAxisTitle("Number of digis per FED channel per event",1);
  theDMBE->cd("Pixel/Endcap");
  meNDigisCOMBEndcap_ = theDMBE->book1D("ALLMODS_ndigisCOMB_Endcap","Number of Digis",200,0.,400.);
  meNDigisCOMBEndcap_->setAxisTitle("Number of digis per module per event",1);
  meNDigisCHANEndcap_ = theDMBE->book1D("ALLMODS_ndigisCHAN_Endcap","Number of Digis",200,0.,400.);
  meNDigisCHANEndcap_->setAxisTitle("Number of digis per FED channel per event",1);
  theDMBE->cd(currDir);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelDigiSource);
