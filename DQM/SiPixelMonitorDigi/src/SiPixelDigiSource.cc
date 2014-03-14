
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
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
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
  bigEventSize( conf_.getUntrackedParameter<int>("bigEventSize",1000) ), 
  isUpgrade( conf_.getUntrackedParameter<bool>("isUpgrade",false) )
{
   //set Token(-s)
   srcToken_ = consumes<edm::DetSetVector<PixelDigi> >(conf_.getParameter<edm::InputTag>( "src" ));

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
  int nTOTmodules;
  if (isUpgrade) { nTOTmodules=1856; } else { nTOTmodules=1440; }
  while(!infile.eof()&&nModsInFile<nTOTmodules) {
    infile >> I_name[nModsInFile] >> I_detId[nModsInFile] >> I_fedId[nModsInFile] >> I_linkId1[nModsInFile] >> I_linkId2[nModsInFile];
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
    for(int i=0; i!=4; i++) nDigisPerDisk[i]=0;  
    nDP1P1M1 = 0;
    nDP1P1M2 = 0;
    nDP1P1M3 = 0;
    nDP1P1M4 = 0;
    nDP1P2M1 = 0;
    nDP1P2M2 = 0;
    nDP1P2M3 = 0;
    nDP2P1M1 = 0;
    nDP2P1M2 = 0;
    nDP2P1M3 = 0;
    nDP2P1M4 = 0;
    nDP2P2M1 = 0;
    nDP2P2M2 = 0;
    nDP2P2M3 = 0;
    nDP3P1M1 = 0;
    nDP3P2M1 = 0;
    nDM1P1M1 = 0;
    nDM1P1M2 = 0;
    nDM1P1M3 = 0;
    nDM1P1M4 = 0;
    nDM1P2M1 = 0;
    nDM1P2M2 = 0;
    nDM1P2M3 = 0;
    nDM2P1M1 = 0;
    nDM2P1M2 = 0;
    nDM2P1M3 = 0;
    nDM2P1M4 = 0;
    nDM2P2M1 = 0;
    nDM2P2M2 = 0;
    nDM2P2M3 = 0;
    nDM3P1M1 = 0;
    nDM3P2M1 = 0;
    nL1M1 = 0;
    nL1M2 = 0;
    nL1M3 = 0;
    nL1M4 = 0;
    nL2M1 = 0;
    nL2M2 = 0;
    nL2M3 = 0;
    nL2M4 = 0;
    nL3M1 = 0;
    nL3M2 = 0;
    nL3M3 = 0;
    nL3M4 = 0;
    nL4M1 = 0;
    nL4M2 = 0;
    nL4M3 = 0;
    nL4M4 = 0;
    
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
  iEvent.getByToken(srcToken_, input);
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
    if (lumiSection%10==0){
      //Now do resets for ROCuppancy maps every 10 ls
      std::string baseDirs[2] = {"Pixel/Barrel", "Pixel/Endcap"};
      for (int i = 0; i < 2; ++i){
	theDMBE->cd(baseDirs[i]);
	vector<string> shellDirs = theDMBE->getSubdirs();
	for (vector<string>::const_iterator it = shellDirs.begin(); it != shellDirs.end(); it++) {
	  theDMBE->cd(*it);
	  vector<string> layDirs = theDMBE->getSubdirs();
	  for (vector<string>::const_iterator itt = layDirs.begin(); itt != layDirs.end(); itt++) {
	    theDMBE->cd(*itt);
	    vector<string> contents = theDMBE->getMEs();
	    for (vector<string>::const_iterator im = contents.begin(); im != contents.end(); im++) {
	      if ((*im).find("rocmap") == string::npos) continue;
	      MonitorElement* me = theDMBE->get((*itt)+"/"+(*im));
	      if(me) me->Reset();}}}}//end for contents//end for layDirs//end for shellDirs//end for bar/EC
    }
  }
  if(!modOn){
    MonitorElement* meReset = theDMBE->get("Pixel/averageDigiOccupancy");
    if(meReset && lumiSection%1==0){
      meReset->Reset();
      nBPIXDigis = 0; 
      nFPIXDigis = 0;
      for(int i=0; i!=40; i++) nDigisPerFed[i]=0;  
    }
  }
  
  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
  for(int i=0; i!=192; i++) numberOfDigis[i]=0;
  for(int i=0; i!=1152; i++) nDigisPerChan[i]=0;  
  for(int i=0; i!=4; i++) nDigisPerDisk[i]=0;  
  for (struct_iter = thePixelStructure.begin() ; struct_iter != thePixelStructure.end() ; struct_iter++) {
    int numberOfDigisMod = (*struct_iter).second->fill(*input, modOn, 
				ladOn, layOn, phiOn, 
				bladeOn, diskOn, ringOn, 
				twoDimOn, reducedSet, twoDimModOn, twoDimOnlyLayDisk,
				nDigisA, nDigisB, isUpgrade);
    if(numberOfDigisMod>0){
      //if((*struct_iter).first == I_detId[39]) 
      //std::cout << "FED " << (*struct_iter).first << " NDigis all modules..." << numberOfDigisMod << std::endl;
      nEventDigis = nEventDigis + numberOfDigisMod;  
      nActiveModules++;  
      bool barrel = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
      bool endcap = DetId((*struct_iter).first).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
      //if((*struct_iter).first >= 302055684 && (*struct_iter).first <= 302197792 ){ // Barrel
      int nBPiXmodules;
      //int nFPixmodules;
      int nTOTmodules;
      if (isUpgrade) {
        nBPiXmodules=1184;
        //nFPixmodules=672;
        nTOTmodules=1856;
      } else {
        nBPiXmodules=768;
        //nFPixmodules=672;
        nTOTmodules=1440;
      }
      if(barrel){ // Barrel
        //cout<<"AAbpix: "<<numberOfDigisMod<<" + "<<nBPIXDigis<<" = ";
        nBPIXDigis = nBPIXDigis + numberOfDigisMod;
	//cout<<nBPIXDigis<<endl;
        for(int i=0; i!=nBPiXmodules; ++i){
          //cout<<"\t\t\t bpix: "<<i<<" , "<<(*struct_iter).first<<" , "<<I_detId[i]<<endl;
          if((*struct_iter).first == I_detId[i]){
	    //if(I_fedId[i]>=32&&I_fedId[i]<=39) std::cout<<"Attention: a BPIX module matched to an FPIX FED!"<<std::endl;
	    nDigisPerFed[I_fedId[i]]=nDigisPerFed[I_fedId[i]]+numberOfDigisMod;
	    //cout<<"BPIX: "<<i<<" , "<<I_fedId[i]<<" , "<<numberOfDigisMod<<" , "<<nDigisPerFed[I_fedId[i]]<<endl;
	    int index1 = 0; int index2 = 0;
	    if(I_linkId1[i]>0) index1 = I_fedId[i]*36+(I_linkId1[i]-1); 
	    if(I_linkId2[i]>0) index2 = I_fedId[i]*36+(I_linkId2[i]-1);
	    if(nDigisA>0 && I_linkId1[i]>0) nDigisPerChan[index1]=nDigisPerChan[index1]+nDigisA;
	    if(nDigisB>0 && I_linkId2[i]>0) nDigisPerChan[index2]=nDigisPerChan[index2]+nDigisB;
	    //if (index1==35 || index2==35) cout<<"BPIX 35: "<<I_detId[i]<<" : "<<I_fedId[i]<<"  "<<I_linkId1[i]<<" , "<<I_fedId[i]<<"  "<<I_linkId2[i]<<" , "<<nDigisA<<" , "<<nDigisB<<endl;
            i=(nBPiXmodules-1);
	  }
        }
      //}else if((*struct_iter).first >= 343999748 && (*struct_iter).first <= 352477708 ){ // Endcap
      }else if(endcap && !isUpgrade){ // Endcap
        //cout<<"AAfpix: "<<nFPIXDigis<<" = ";
        nFPIXDigis = nFPIXDigis + numberOfDigisMod;
	//cout<<nFPIXDigis<<endl;
        PixelEndcapName::HalfCylinder side = PixelEndcapName(DetId((*struct_iter).first)).halfCylinder();
	int disk = PixelEndcapName(DetId((*struct_iter).first)).diskName();
	int blade = PixelEndcapName(DetId((*struct_iter).first)).bladeName();
        int panel = PixelEndcapName(DetId((*struct_iter).first)).pannelName();
        int module = PixelEndcapName(DetId((*struct_iter).first)).plaquetteName();
	//std::cout<<"Endcap: "<<side<<" , "<<disk<<" , "<<blade<<" , "<<panel<<" , "<<std::endl;
	int iter=0; int i=0;
	if(side==PixelEndcapName::mI){
	  if(disk==1){
	    i=0;
	    if(panel==1){ if(module==1) nDM1P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDM1P1M2+=numberOfDigisMod; 
			  else if(module==3) nDM1P1M3+=numberOfDigisMod; 
			  else if(module==4) nDM1P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDM1P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDM1P2M2+=numberOfDigisMod; 
			       else if(module==3) nDM1P2M3+=numberOfDigisMod; }
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
	    if(panel==1){ if(module==1) nDM2P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDM2P1M2+=numberOfDigisMod; 
			  else if(module==3) nDM2P1M3+=numberOfDigisMod; 
			  else if(module==4) nDM2P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDM2P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDM2P2M2+=numberOfDigisMod; 
			       else if(module==3) nDM2P2M3+=numberOfDigisMod; }
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
	    if(panel==1){ if(module==1) nDM1P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDM1P1M2+=numberOfDigisMod; 
			  else if(module==3) nDM1P1M3+=numberOfDigisMod; 
			  else if(module==4) nDM1P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDM1P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDM1P2M2+=numberOfDigisMod; 
			       else if(module==3) nDM1P2M3+=numberOfDigisMod; }
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
	    if(panel==1){ if(module==1) nDM2P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDM2P1M2+=numberOfDigisMod; 
			  else if(module==3) nDM2P1M3+=numberOfDigisMod; 
			  else if(module==4) nDM2P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDM2P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDM2P2M2+=numberOfDigisMod; 
			       else if(module==3) nDM2P2M3+=numberOfDigisMod; }
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
	    if(panel==1){ if(module==1) nDP1P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDP1P1M2+=numberOfDigisMod; 
			  else if(module==3) nDP1P1M3+=numberOfDigisMod; 
			  else if(module==4) nDP1P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDP1P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDP1P2M2+=numberOfDigisMod; 
			       else if(module==3) nDP1P2M3+=numberOfDigisMod; }
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
	    if(panel==1){ if(module==1) nDP2P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDP2P1M2+=numberOfDigisMod; 
			  else if(module==3) nDP2P1M3+=numberOfDigisMod; 
			  else if(module==4) nDP2P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDP2P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDP2P2M2+=numberOfDigisMod; 
			       else if(module==3) nDP2P2M3+=numberOfDigisMod; }
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
	    i=144;
	    if(panel==1){ if(module==1) nDP1P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDP1P1M2+=numberOfDigisMod; 
			  else if(module==3) nDP1P1M3+=numberOfDigisMod; 
			  else if(module==4) nDP1P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDP1P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDP1P2M2+=numberOfDigisMod; 
			       else if(module==3) nDP1P2M3+=numberOfDigisMod; }
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
	    i=168;
	    if(panel==1){ if(module==1) nDP2P1M1+=numberOfDigisMod; 
	                  else if(module==2) nDP2P1M2+=numberOfDigisMod; 
			  else if(module==3) nDP2P1M3+=numberOfDigisMod; 
			  else if(module==4) nDP2P1M4+=numberOfDigisMod;}
	    else if(panel==2){ if(module==1) nDP2P2M1+=numberOfDigisMod; 
	                       else if(module==2) nDP2P2M2+=numberOfDigisMod; 
			       else if(module==3) nDP2P2M3+=numberOfDigisMod; }
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
	numberOfDigis[iter]=numberOfDigis[iter]+numberOfDigisMod;
	//if(side==PixelEndcapName::pO||side==PixelEndcapName::pI){
	//  if(disk==2){ 
	//  std::cout<<"status: "<<iter<<","<<side<<","<<disk<<","<<blade<<","<<panel<<","<<numberOfDigisMod<<","<<numberOfDigis[iter]<<std::endl;       
        //}}
        for(int i=nBPiXmodules; i!=nTOTmodules; i++){ 
          //cout<<"\t\t\t fpix: "<<i<<" , "<<(*struct_iter).first<<" , "<<I_detId[i]<<endl;
          if((*struct_iter).first == I_detId[i]){
	    //if(I_fedId[i]<32||I_fedId[i]>39) std::cout<<"Attention: an FPIX module matched to a BPIX FED!"<<std::endl;
	    nDigisPerFed[I_fedId[i]]=nDigisPerFed[I_fedId[i]]+numberOfDigisMod;
	    //cout<<"FPIX: "<<i<<" , "<<I_fedId[i]<<" , "<<nDigisPerFed[I_fedId[i]]<< ", "<<numberOfDigisMod << endl;
            i=nTOTmodules-1;
	  }
        }
	//cout<<"NDigis Endcap: "<<nDM1P1M1/2.<<" "<<nDM1P2M1/6.<<" "<<nDM1P1M2/6.<<" "<<nDM1P2M2/8.<<" "<<nDM1P1M3/8.<<" "<<nDM1P2M3/10.<<" "<<nDM1P1M4/5.<<endl;
      } //endif Barrel/(Endcap && !isUpgrade)
      else if (endcap && isUpgrade) {
        //cout<<"AAfpix: "<<nFPIXDigis<<" = ";
        nFPIXDigis = nFPIXDigis + numberOfDigisMod;
        //cout<<nFPIXDigis<<endl;
        PixelEndcapNameUpgrade::HalfCylinder side = PixelEndcapNameUpgrade(DetId((*struct_iter).first)).halfCylinder();
        int disk = PixelEndcapNameUpgrade(DetId((*struct_iter).first)).diskName();
        int blade = PixelEndcapNameUpgrade(DetId((*struct_iter).first)).bladeName();
        int panel = PixelEndcapNameUpgrade(DetId((*struct_iter).first)).pannelName();
        int module = PixelEndcapNameUpgrade(DetId((*struct_iter).first)).plaquetteName();
        
        int iter=0; int i=0;
        if(side==PixelEndcapNameUpgrade::mI){
          if(disk==1){
            i=0;
            if(panel==1){ if(module==1) nDM1P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDM1P2M1+=numberOfDigisMod; }
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
          }else if(disk==2){
            i=22;
            if(panel==1){ if(module==1) nDM2P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDM2P2M1+=numberOfDigisMod; }
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
          }else if(disk==3){
            i=44;
            if(panel==1){ if(module==1) nDM3P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDM3P2M1+=numberOfDigisMod; }
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
          }
        }else if(side==PixelEndcapNameUpgrade::mO){
          if(disk==1){
            i=66;
            if(panel==1){ if(module==1) nDM1P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDM1P2M1+=numberOfDigisMod; }
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
            if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
            if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
            if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
            if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
            if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }else if(disk==2){
           i=100;
           if(panel==1){ if(module==1) nDM2P1M1+=numberOfDigisMod; }
           else if(panel==2){ if(module==1) nDM2P2M1+=numberOfDigisMod; }
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
           if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
           if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
           if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
           if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
           if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }else if (disk==3){
            i=134;
            if(panel==1){ if(module==1) nDM3P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDM3P2M1+=numberOfDigisMod; }
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
            if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
            if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
            if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
            if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
            if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }
        }else if(side==PixelEndcapNameUpgrade::pI){
          if(disk==1){
            i=168;
            if(panel==1){ if(module==1) nDP1P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP1P2M1+=numberOfDigisMod; }
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
          }else if(disk==2){
            i=190;
            if(panel==1){ if(module==1) nDP2P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP2P2M1+=numberOfDigisMod; }
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
          }else if(disk==3){
            i=212;
            if(panel==1){ if(module==1) nDP3P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP3P2M1+=numberOfDigisMod; }
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
          }
        }else if(side==PixelEndcapNameUpgrade::pO){
          if(disk==1){
            i=234;
            if(panel==1){ if(module==1) nDP1P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP1P2M1+=numberOfDigisMod; }
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
            if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
            if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
            if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
            if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
            if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }else if(disk==2){
            i=268;
            if(panel==1){ if(module==1) nDP2P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP2P2M1+=numberOfDigisMod; }
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
            if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
            if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
            if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
            if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
            if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }else if(disk==3){
            i=302;
            if(panel==1){ if(module==1) nDP3P1M1+=numberOfDigisMod; }
            else if(panel==2){ if(module==1) nDP3P2M1+=numberOfDigisMod; }
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
            if(blade==13){ if(panel==1) iter=i+24; else if(panel==2) iter=i+25; }
            if(blade==14){ if(panel==1) iter=i+26; else if(panel==2) iter=i+27; }
            if(blade==15){ if(panel==1) iter=i+28; else if(panel==2) iter=i+29; }
            if(blade==16){ if(panel==1) iter=i+30; else if(panel==2) iter=i+31; }
            if(blade==17){ if(panel==1) iter=i+32; else if(panel==2) iter=i+33; }
          }
        }
        numberOfDigis[iter]=numberOfDigis[iter]+numberOfDigisMod;
        //if(side==PixelEndcapNameUpgrade::pO||side==PixelEndcapNameUpgrade::pI){
        //  if(disk==2){ 
        //  std::cout<<"status: "<<iter<<","<<side<<","<<disk<<","<<blade<<","<<panel<<","<<numberOfDigisMod<<","<<numberOfDigis[iter]<<std::endl;       
        //}}
        for(int i=nBPiXmodules; i!=nTOTmodules; i++){
          //cout<<"\t\t\t fpix: "<<i<<" , "<<(*struct_iter).first<<" , "<<I_detId[i]<<endl;
          if((*struct_iter).first == I_detId[i]){
	    //if(I_fedId[i]<32||I_fedId[i]>39) std::cout<<"Attention: an FPIX module matched to a BPIX FED!"<<std::endl;
	    nDigisPerFed[I_fedId[i]]=nDigisPerFed[I_fedId[i]]+numberOfDigisMod;
	    //cout<<"FPIX: "<<i<<" , "<<I_fedId[i]<<" , "<<nDigisPerFed[I_fedId[i]]<< ", "<<numberOfDigisMod << endl;
            i=nTOTmodules-1;
	  }
        }
	//cout<<"NDigis Endcap: "<<nDM1P1M1/2.<<" "<<nDM1P2M1/6.<<" "<<nDM1P1M2/6.<<" "<<nDM1P2M2/8.<<" "<<nDM1P1M3/8.<<" "<<nDM1P2M3/10.<<" "<<nDM1P1M4/5.<<endl;
      }//endif(Endcap && isUpgrade)
      //cout<<"numberOfDigis: "<<numberOfDigisMod<<" , nBPIXDigis: "<<nBPIXDigis<<" , nFPIXDigis: "<<nFPIXDigis<<endl;
      // digi occupancy per individual FED channel:
    } // endif any digis in this module
  } // endfor loop over all modules

  //A really, really ugly way to do the occupancy-based 
  int NzeroROCs[2]        = {0,-672};
  int NloEffROCs[2]       = {0,-672};
  std::string baseDirs[2] = {"Pixel/Barrel", "Pixel/Endcap"};
  if (lumiSection%10> 2){
    for (int i = 0; i < 2; ++i){
      theDMBE->cd(baseDirs[i]);
      vector<string> shellDirs = theDMBE->getSubdirs();
      for (vector<string>::const_iterator it = shellDirs.begin(); it != shellDirs.end(); it++) {
	theDMBE->cd(*it);
	vector<string> layDirs = theDMBE->getSubdirs();
	for (vector<string>::const_iterator itt = layDirs.begin(); itt != layDirs.end(); itt++) {
	  theDMBE->cd(*itt);
	  vector<string> contents = theDMBE->getMEs(); 
	  for (vector<string>::const_iterator im = contents.begin(); im != contents.end(); im++) {
	    if ((*im).find("rocmap") == string::npos) continue;
	    MonitorElement* me  = theDMBE->get((*itt)+"/"+(*im));
	    if(!me) continue;
	    MonitorElement* me2;
	    me2 = theDMBE->get((*itt)+"/zeroOccROC_map");
	    float SF = 1.0; if (me->getEntries() > 0) SF = float(me->getNbinsX()*me->getNbinsY()/me->getEntries());
	    for (int ii = 1; ii < me->getNbinsX()+1; ++ii){for (int jj = 1; jj < me->getNbinsY()+1; ++jj){
		//Putting in conversion to layer maps.. again, ugly way to do it...
		float localX = float(ii)-0.5;
		float localY = float(jj)/2.0 + 1.25;
		if (i ==1) localY = float(jj)/2.0 + 0.75;
		if (me->getBinContent(ii,jj)    <   1) {++NzeroROCs[i]; if (me2) me2->Fill(localX, localY);}
		if (me->getBinContent(ii,jj)*SF < 0.25) ++NloEffROCs[i];}}
	  }
	}
      }
    }
    for (int i =0; i < 2; ++i) NloEffROCs[i] = NloEffROCs[i] - NzeroROCs[i];
    MonitorElement* menoOcc=theDMBE->get("Pixel/noOccROCsBarrel");
    MonitorElement* meloOcc=theDMBE->get("Pixel/loOccROCsBarrel");
    if(menoOcc) menoOcc->setBinContent(1+lumiSection/10, NzeroROCs[0]);
    if(meloOcc) meloOcc->setBinContent(1+lumiSection/10, NloEffROCs[0]);
    MonitorElement* menoOcc1=theDMBE->get("Pixel/noOccROCsEndcap");
    MonitorElement* meloOcc1=theDMBE->get("Pixel/loOccROCsEndcap");
    if(menoOcc1) menoOcc1->setBinContent(1+lumiSection/10, NzeroROCs[1]);
    if(meloOcc1) meloOcc1->setBinContent(1+lumiSection/10, NloEffROCs[1]);
    theDMBE->cd();
  }
//  if(lumiSection>lumSec){ lumSec = lumiSection; nLumiSecs++; }
//  if(nEventDigis>bigEventSize) nBigEvents++;
//  if(nLumiSecs%5==0){
  
  MonitorElement* meE; MonitorElement* meE1; MonitorElement* meE2; MonitorElement* meE3; MonitorElement* meE4; 
  MonitorElement* meE5; MonitorElement* meE6;
  if (!isUpgrade) {
  meE=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_Endcap");
  if(meE){ for(int j=0; j!=192; j++) if(numberOfDigis[j]>0) meE->Fill((float)numberOfDigis[j]);}
  meE1=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDm1");
  if(meE1){ for(int j=0; j!=72; j++) if((j<24||j>47)&&numberOfDigis[j]>0) meE1->Fill((float)numberOfDigis[j]);}
  meE2=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDm2");
  if(meE2){ for(int j=24; j!=96; j++) if((j<48||j>71)&&numberOfDigis[j]>0) meE2->Fill((float)numberOfDigis[j]);}
  meE3=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDp1");
  if(meE3){ for(int j=96; j!=168; j++) if((j<120||j>143)&&numberOfDigis[j]>0) meE3->Fill((float)numberOfDigis[j]);}
  meE4=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDp2");
  if(meE4){ for(int j=120; j!=192; j++) if((j<144||j>167)&&numberOfDigis[j]>0) meE4->Fill((float)numberOfDigis[j]);}
  } else if (isUpgrade) {
    meE=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_Endcap");
    if(meE){ for(int j=0; j!=336; j++) if(numberOfDigis[j]>0) meE->Fill((float)numberOfDigis[j]);}
    meE1=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDm1");
    if(meE1){ for(int j=0; j!=100; j++) if((j<22||j>65)&&numberOfDigis[j]>0) meE1->Fill((float)numberOfDigis[j]);}
    meE2=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDm2");
    if(meE2){ for(int j=22; j!=134; j++) if((j<44||j>99)&&numberOfDigis[j]>0) meE2->Fill((float)numberOfDigis[j]);}
    meE3=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDm3");
    if(meE3){ for(int j=44; j!=168; j++) if((j<66||j>133)&&numberOfDigis[j]>0) meE3->Fill((float)numberOfDigis[j]);}
    meE4=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDp1");
    if(meE4){ for(int j=168; j!=268; j++) if((j<190||j>233)&&numberOfDigis[j]>0) meE4->Fill((float)numberOfDigis[j]);}
    meE5=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDp2");
    if(meE5){ for(int j=190; j!=302; j++) if((j<212||j>267)&&numberOfDigis[j]>0) meE5->Fill((float)numberOfDigis[j]);}
    meE6=theDMBE->get("Pixel/Endcap/ALLMODS_ndigisCHAN_EndcapDp3");
    if(meE6){ for(int j=212; j!=336; j++) if((j<234||j>301)&&numberOfDigis[j]>0) meE6->Fill((float)numberOfDigis[j]);}
  }
  
  MonitorElement* me1; MonitorElement* me2; MonitorElement* me3; MonitorElement* me4; MonitorElement* me5;
  MonitorElement* me6; MonitorElement* me7; MonitorElement* me8; MonitorElement* me9; MonitorElement* me10; MonitorElement* me11;
  MonitorElement* me12; MonitorElement* me13; MonitorElement* me14; MonitorElement* me15; MonitorElement* me16; MonitorElement* me17;
  MonitorElement* me18; MonitorElement* me19; MonitorElement* me20; MonitorElement* me21; MonitorElement* me22; MonitorElement* me23;
  MonitorElement* me24; MonitorElement* me25; MonitorElement* me26; MonitorElement* me27; MonitorElement* me28; MonitorElement* me29;
  MonitorElement* me30; MonitorElement* me31; MonitorElement* me32; MonitorElement* me33; MonitorElement* me34; MonitorElement* me35;
  MonitorElement* me36;
  me1=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh1");
  if(me1){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+0]>0) me1->Fill((float)nDigisPerChan[i*36+0]);}
  me2=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh2");
  if(me2){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+1]>0) me2->Fill((float)nDigisPerChan[i*36+1]);}
  me3=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh3");
  if(me3){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+2]>0) me3->Fill((float)nDigisPerChan[i*36+2]);}
  me4=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh4");
  if(me4){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+3]>0) me4->Fill((float)nDigisPerChan[i*36+3]);}
  me5=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh5");
  if(me5){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+4]>0) me5->Fill((float)nDigisPerChan[i*36+4]);}
  me6=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh6");
  if(me6){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+5]>0) me6->Fill((float)nDigisPerChan[i*36+5]);}
  me7=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh7");
  if(me7){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+6]>0) me7->Fill((float)nDigisPerChan[i*36+6]);}
  me8=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh8");
  if(me8){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+7]>0) me8->Fill((float)nDigisPerChan[i*36+7]);}
  me9=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh9");
  if(me9){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+8]>0) me9->Fill((float)nDigisPerChan[i*36+8]);}
  me10=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh10");
  if(me10){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+9]>0) me10->Fill((float)nDigisPerChan[i*36+9]);}
  me11=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh11");
  if(me11){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+10]>0) me11->Fill((float)nDigisPerChan[i*36+10]);}
  me12=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh12");
  if(me12){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+11]>0) me12->Fill((float)nDigisPerChan[i*36+11]);}
  me13=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh13");
  if(me13){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+12]>0) me13->Fill((float)nDigisPerChan[i*36+12]);}
  me14=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh14");
  if(me14){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+13]>0) me14->Fill((float)nDigisPerChan[i*36+13]);}
  me15=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh15");
  if(me15){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+14]>0) me15->Fill((float)nDigisPerChan[i*36+14]);}
  me16=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh16");
  if(me16){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+15]>0) me16->Fill((float)nDigisPerChan[i*36+15]);}
  me17=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh17");
  if(me17){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+16]>0) me17->Fill((float)nDigisPerChan[i*36+16]);}
  me18=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh18");
  if(me18){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+17]>0) me18->Fill((float)nDigisPerChan[i*36+17]);}
  me19=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh19");
  if(me19){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+18]>0) me19->Fill((float)nDigisPerChan[i*36+18]);}
  me20=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh20");
  if(me20){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+19]>0) me20->Fill((float)nDigisPerChan[i*36+19]);}
  me21=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh21");
  if(me21){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+20]>0) me21->Fill((float)nDigisPerChan[i*36+20]);}
  me22=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh22");
  if(me22){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+21]>0) me22->Fill((float)nDigisPerChan[i*36+21]);}
  me23=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh23");
  if(me23){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+22]>0) me23->Fill((float)nDigisPerChan[i*36+22]);}
  me24=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh24");
  if(me24){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+23]>0) me24->Fill((float)nDigisPerChan[i*36+23]);}
  me25=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh25");
  if(me25){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+24]>0) me25->Fill((float)nDigisPerChan[i*36+24]);}
  me26=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh26");
  if(me26){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+25]>0) me26->Fill((float)nDigisPerChan[i*36+25]);}
  me27=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh27");
  if(me27){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+26]>0) me27->Fill((float)nDigisPerChan[i*36+26]);}
  me28=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh28");
  if(me28){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+27]>0) me28->Fill((float)nDigisPerChan[i*36+27]);}
  me29=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh29");
  if(me29){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+28]>0) me29->Fill((float)nDigisPerChan[i*36+28]);}
  me30=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh30");
  if(me30){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+29]>0) me30->Fill((float)nDigisPerChan[i*36+29]);}
  me31=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh31");
  if(me31){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+30]>0) me31->Fill((float)nDigisPerChan[i*36+30]);}
  me32=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh32");
  if(me32){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+31]>0) me32->Fill((float)nDigisPerChan[i*36+31]);}
  me33=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh33");
  if(me33){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+32]>0) me33->Fill((float)nDigisPerChan[i*36+32]);}
  me34=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh34");
  if(me34){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+33]>0) me34->Fill((float)nDigisPerChan[i*36+33]);}
  me35=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh35");
  if(me35){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+34]>0) me35->Fill((float)nDigisPerChan[i*36+34]);}
  me36=theDMBE->get("Pixel/Barrel/ALLMODS_ndigisCHAN_BarrelCh36");
  if(me36){ for(int i=0; i!=32; i++) if(nDigisPerChan[i*36+35]>0) me36->Fill((float)nDigisPerChan[i*36+35]);}
  
  // Rate of events with >N digis:
  MonitorElement* meX1;
  if(nEventDigis>bigEventSize){
    meX1 = theDMBE->get("Pixel/bigEventRate");
    if(meX1) meX1->Fill(lumiSection,1./23.);    
  }
  //std::cout<<"nEventDigis: "<<nEventDigis<<" , nLumiSecs: "<<nLumiSecs<<" , nBigEvents: "<<nBigEvents<<std::endl;
  
  // Rate of pixel events and total number of pixel events per BX:
  MonitorElement* meX2; MonitorElement* meX3;
  if(nActiveModules>=4){
    meX2 = theDMBE->get("Pixel/pixEvtsPerBX");
    if(meX2) meX2->Fill(float(bx));
    meX3 = theDMBE->get("Pixel/pixEventRate");
    if(meX3) meX3->Fill(lumiSection, 1./23.);
  }
  
  // Actual digi occupancy in a FED compared to average digi occupancy per FED
  MonitorElement* meX4; MonitorElement* meX5;
  meX4 = theDMBE->get("Pixel/averageDigiOccupancy");
  meX5 = theDMBE->get("Pixel/avgfedDigiOccvsLumi");
  if(meX4){
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
      meX4->setBinContent(i+1,averageOcc);
      int lumiSections8 = int(lumiSection/8);
      if (modOn){
	if (meX5){
	  meX5->setBinContent(1+lumiSections8, i+1, averageOcc);
	}//endif meX5
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

      }	else if((detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (!isUpgrade)) {
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
      }	else if( (detId.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) && (isUpgrade)) {
	LogDebug ("PixelDQM") << " ---> Adding Endcap Module " <<  detId.rawId() << endl;
	uint32_t id = detId();
	SiPixelDigiModule* theModule = new SiPixelDigiModule(id, ncols, nrows);
        
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
	
	thePixelStructure.insert(pair<uint32_t,SiPixelDigiModule*> (id,theModule));
      }//end_elseif(isUpgrade)

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
  char title[80];   sprintf(title, "Rate of events with >%i digis;LumiSection;Rate [Hz]",bigEventSize);
  bigEventRate    = theDMBE->book1D("bigEventRate",title,5000,0.,5000.);
  char title1[80];  sprintf(title1, "Pixel events vs. BX;BX;# events");
  pixEvtsPerBX    = theDMBE->book1D("pixEvtsPerBX",title1,3565,0.,3565.);
  char title2[80];  sprintf(title2, "Rate of Pixel events;LumiSection;Rate [Hz]");
  pixEventRate    = theDMBE->book1D("pixEventRate",title2,5000,0.,5000.);
  char title3[80];  sprintf(title3, "Number of Zero-Occupancy Barrel ROCs;LumiSection;N_{ZERO-OCCUPANCY} Barrel ROCs");
  noOccROCsBarrel = theDMBE->book1D("noOccROCsBarrel",title3,500,0.,5000.);
  char title4[80];  sprintf(title4, "Number of Low-Efficiency Barrel ROCs;LumiSection;N_{LO EFF} Barrel ROCs");
  loOccROCsBarrel = theDMBE->book1D("loOccROCsBarrel",title4,500,0.,5000.);
  char title5[80];  sprintf(title5, "Number of Zero-Occupancy Endcap ROCs;LumiSection;N_{ZERO-OCCUPANCY} Endcap ROCs");
  noOccROCsEndcap = theDMBE->book1D("noOccROCsEndcap",title5,500,0.,5000.);
  char title6[80];  sprintf(title6, "Number of Low-Efficiency Endcap ROCs;LumiSection;N_{LO EFF} Endcap ROCs");
  loOccROCsEndcap = theDMBE->book1D("loOccROCsEndcap",title6,500,0.,5000.);
  char title7[80];  sprintf(title7, "Average digi occupancy per FED;FED;NDigis/<NDigis>");
  averageDigiOccupancy = theDMBE->book1D("averageDigiOccupancy",title7,40,-0.5,39.5);
  averageDigiOccupancy->setLumiFlag();
  if(modOn){
    char title4[80]; sprintf(title4, "FED Digi Occupancy (NDigis/<NDigis>) vs LumiSections;Lumi Section;FED");
    avgfedDigiOccvsLumi = theDMBE->book2D ("avgfedDigiOccvsLumi", title4, 400,0., 3200., 40, -0.5, 39.5);
  }  
  std::map<uint32_t,SiPixelDigiModule*>::iterator struct_iter;
 
  SiPixelFolderOrganizer theSiPixelFolder;

  for(struct_iter = thePixelStructure.begin(); struct_iter != thePixelStructure.end(); struct_iter++){
    /// Create folder tree and book histograms 
    if(modOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,0,isUpgrade)){
	(*struct_iter).second->book( conf_,0,twoDimOn,hiRes, reducedSet, twoDimModOn, isUpgrade);
      } else {

	if(!isPIB) throw cms::Exception("LogicError")
	  << "[SiPixelDigiSource::bookMEs] Creation of DQM folder failed";
      }
    }
    if(ladOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,1,isUpgrade)){
	(*struct_iter).second->book( conf_,1,twoDimOn,hiRes, reducedSet, isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LADDER-FOLDER\n";
      }
   
    }
    if(layOn || twoDimOnlyLayDisk){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,2,isUpgrade)){
	(*struct_iter).second->book( conf_,2,twoDimOn,hiRes, reducedSet, twoDimOnlyLayDisk, isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH LAYER-FOLDER\n";
      }
    }

    if(phiOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,3,isUpgrade)){
	(*struct_iter).second->book( conf_,3,twoDimOn,hiRes, reducedSet, isUpgrade);
	} else {
        LogDebug ("PixelDQM") << "PROBLEM WITH PHI-FOLDER\n";
      }
    }
    if(bladeOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,4,isUpgrade)){
	(*struct_iter).second->book( conf_,4,twoDimOn,hiRes, reducedSet, isUpgrade);
	} else {
	LogDebug ("PixelDQM") << "PROBLEM WITH BLADE-FOLDER\n";
      }
    }
    if(diskOn || twoDimOnlyLayDisk){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,5,isUpgrade)){
	(*struct_iter).second->book( conf_,5,twoDimOn,hiRes, reducedSet, twoDimOnlyLayDisk, isUpgrade);
      } else {
	LogDebug ("PixelDQM") << "PROBLEM WITH DISK-FOLDER\n";
      }
    }
    if(ringOn){
      if(theSiPixelFolder.setModuleFolder((*struct_iter).first,6,isUpgrade)){
	(*struct_iter).second->book( conf_,6,twoDimOn,hiRes, reducedSet, isUpgrade);
      } else {
	LogDebug ("PixelDQM") << "PROBLEM WITH RING-FOLDER\n";
      }
    }
  }
  std::string currDir = theDMBE->pwd();
  theDMBE->cd("Pixel/Barrel");
  meNDigisCOMBBarrel_ = theDMBE->book1D("ALLMODS_ndigisCOMB_Barrel","Number of Digis",200,0.,400.);
  meNDigisCOMBBarrel_->setAxisTitle("Number of digis per module per event",1);
  meNDigisCHANBarrel_ = theDMBE->book1D("ALLMODS_ndigisCHAN_Barrel","Number of Digis",100,0.,1000.);
  meNDigisCHANBarrel_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelL1_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelL1","Number of Digis L1",100,0.,1000.);
  meNDigisCHANBarrelL1_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelL2_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelL2","Number of Digis L2",100,0.,1000.);
  meNDigisCHANBarrelL2_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelL3_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelL3","Number of Digis L3",100,0.,1000.);
  meNDigisCHANBarrelL3_->setAxisTitle("Number of digis per FED channel per event",1);
  if (isUpgrade) {
    meNDigisCHANBarrelL4_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelL4","Number of Digis L4",100,0.,1000.);
    meNDigisCHANBarrelL4_->setAxisTitle("Number of digis per FED channel per event",1);
  }
  meNDigisCHANBarrelCh1_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh1","Number of Digis Ch1",100,0.,1000.);
  meNDigisCHANBarrelCh1_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh2_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh2","Number of Digis Ch2",100,0.,1000.);
  meNDigisCHANBarrelCh2_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh3_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh3","Number of Digis Ch3",100,0.,1000.);
  meNDigisCHANBarrelCh3_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh4_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh4","Number of Digis Ch4",100,0.,1000.);
  meNDigisCHANBarrelCh4_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh5_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh5","Number of Digis Ch5",100,0.,1000.);
  meNDigisCHANBarrelCh5_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh6_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh6","Number of Digis Ch6",100,0.,1000.);
  meNDigisCHANBarrelCh6_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh7_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh7","Number of Digis Ch7",100,0.,1000.);
  meNDigisCHANBarrelCh7_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh8_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh8","Number of Digis Ch8",100,0.,1000.);
  meNDigisCHANBarrelCh8_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh9_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh9","Number of Digis Ch9",100,0.,1000.);
  meNDigisCHANBarrelCh9_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh10_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh10","Number of Digis Ch10",100,0.,1000.);
  meNDigisCHANBarrelCh10_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh11_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh11","Number of Digis Ch11",100,0.,1000.);
  meNDigisCHANBarrelCh11_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh12_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh12","Number of Digis Ch12",100,0.,1000.);
  meNDigisCHANBarrelCh12_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh13_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh13","Number of Digis Ch13",100,0.,1000.);
  meNDigisCHANBarrelCh13_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh14_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh14","Number of Digis Ch14",100,0.,1000.);
  meNDigisCHANBarrelCh14_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh15_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh15","Number of Digis Ch15",100,0.,1000.);
  meNDigisCHANBarrelCh15_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh16_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh16","Number of Digis Ch16",100,0.,1000.);
  meNDigisCHANBarrelCh16_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh17_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh17","Number of Digis Ch17",100,0.,1000.);
  meNDigisCHANBarrelCh17_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh18_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh18","Number of Digis Ch18",100,0.,1000.);
  meNDigisCHANBarrelCh18_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh19_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh19","Number of Digis Ch19",100,0.,1000.);
  meNDigisCHANBarrelCh19_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh20_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh20","Number of Digis Ch20",100,0.,1000.);
  meNDigisCHANBarrelCh20_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh21_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh21","Number of Digis Ch21",100,0.,1000.);
  meNDigisCHANBarrelCh21_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh22_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh22","Number of Digis Ch22",100,0.,1000.);
  meNDigisCHANBarrelCh22_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh23_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh23","Number of Digis Ch23",100,0.,1000.);
  meNDigisCHANBarrelCh23_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh24_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh24","Number of Digis Ch24",100,0.,1000.);
  meNDigisCHANBarrelCh24_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh25_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh25","Number of Digis Ch25",100,0.,1000.);
  meNDigisCHANBarrelCh25_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh26_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh26","Number of Digis Ch26",100,0.,1000.);
  meNDigisCHANBarrelCh26_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh27_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh27","Number of Digis Ch27",100,0.,1000.);
  meNDigisCHANBarrelCh27_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh28_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh28","Number of Digis Ch28",100,0.,1000.);
  meNDigisCHANBarrelCh28_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh29_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh29","Number of Digis Ch29",100,0.,1000.);
  meNDigisCHANBarrelCh29_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh30_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh30","Number of Digis Ch30",100,0.,1000.);
  meNDigisCHANBarrelCh30_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh31_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh31","Number of Digis Ch31",100,0.,1000.);
  meNDigisCHANBarrelCh31_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh32_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh32","Number of Digis Ch32",100,0.,1000.);
  meNDigisCHANBarrelCh32_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh33_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh33","Number of Digis Ch33",100,0.,1000.);
  meNDigisCHANBarrelCh33_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh34_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh34","Number of Digis Ch34",100,0.,1000.);
  meNDigisCHANBarrelCh34_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh35_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh35","Number of Digis Ch35",100,0.,1000.);
  meNDigisCHANBarrelCh35_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANBarrelCh36_ = theDMBE->book1D("ALLMODS_ndigisCHAN_BarrelCh36","Number of Digis Ch36",100,0.,1000.);
  meNDigisCHANBarrelCh36_->setAxisTitle("Number of digis per FED channel per event",1);
  theDMBE->cd("Pixel/Endcap");
  meNDigisCOMBEndcap_ = theDMBE->book1D("ALLMODS_ndigisCOMB_Endcap","Number of Digis",200,0.,400.);
  meNDigisCOMBEndcap_->setAxisTitle("Number of digis per module per event",1);
  meNDigisCHANEndcap_ = theDMBE->book1D("ALLMODS_ndigisCHAN_Endcap","Number of Digis",100,0.,1000.);
  meNDigisCHANEndcap_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANEndcapDp1_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDp1","Number of Digis Disk p1",100,0.,1000.);
  meNDigisCHANEndcapDp1_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANEndcapDp2_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDp2","Number of Digis Disk p2",100,0.,1000.);
  meNDigisCHANEndcapDp2_->setAxisTitle("Number of digis per FED channel per event",1);
  if (isUpgrade) {
    meNDigisCHANEndcapDp3_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDp3","Number of Digis Disk p3",100,0.,1000.);
    meNDigisCHANEndcapDp3_->setAxisTitle("Number of digis per FED channel per event",1);
  }
  meNDigisCHANEndcapDm1_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDm1","Number of Digis Disk m1",100,0.,1000.);
  meNDigisCHANEndcapDm1_->setAxisTitle("Number of digis per FED channel per event",1);
  meNDigisCHANEndcapDm2_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDm2","Number of Digis Disk m2",100,0.,1000.);
  meNDigisCHANEndcapDm2_->setAxisTitle("Number of digis per FED channel per event",1);
  if (isUpgrade) {
    meNDigisCHANEndcapDm3_ = theDMBE->book1D("ALLMODS_ndigisCHAN_EndcapDm3","Number of Digis Disk m3",100,0.,1000.);
    meNDigisCHANEndcapDm3_->setAxisTitle("Number of digis per FED channel per event",1);
  }
  theDMBE->cd(currDir);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelDigiSource);
