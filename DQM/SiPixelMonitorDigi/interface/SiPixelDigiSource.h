#ifndef SiPixelMonitorDigi_SiPixelDigiSource_h
#define SiPixelMonitorDigi_SiPixelDigiSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorDigi
// Class  :     SiPixelDigiSource
// 
/**

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiSource.h,v 1.29 2013/04/17 09:48:23 itopsisg Exp $
//

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>

 class SiPixelDigiSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelDigiSource(const edm::ParameterSet& conf);
       ~SiPixelDigiSource();

       typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;
       virtual void beginRun(const edm::Run&, edm::EventSetup const&) ;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs();

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       bool saveFile;
       bool isPIB;
       bool slowDown;
       bool modOn; 
       bool twoDimOn;
       bool twoDimModOn; 
       bool twoDimOnlyLayDisk;
       bool hiRes;
       bool reducedSet;
       //barrel:
       bool ladOn, layOn, phiOn;
       //forward:
       bool ringOn, bladeOn, diskOn; 
       int eventNo;
       int lumSec;
       int nLumiSecs;
       DQMStore* theDMBE;
       std::map<uint32_t,SiPixelDigiModule*> thePixelStructure;

       int bigEventSize;
       bool isUpgrade;
       
       bool firstRun;
       
/*       std::string I_name[1440];
       unsigned int I_detId[1440];
       int I_fedId[1440];
       int I_linkId1[1440];
       int I_linkId2[1440];
       int nDigisPerFed[40];
       int nDigisPerChan[1152];
       int nDigisPerDisk[4];
       int numberOfDigis[192];
       int nDigisA;
       int nDigisB;
*/

       std::string I_name[1856];
       unsigned int I_detId[1856];
       int I_fedId[1856];
       int I_linkId1[1856];
       int I_linkId2[1856];
       int nDigisPerFed[40];
       int nDigisPerChan[1152];
       int nDigisPerDisk[6];
       int numberOfDigis[336];
       int nDigisA;
       int nDigisB;

       int nDP1P1M1;
       int nDP1P1M2;
       int nDP1P1M3;
       int nDP1P1M4;
       int nDP1P2M1;
       int nDP1P2M2;
       int nDP1P2M3;
       int nDP2P1M1;
       int nDP2P1M2;
       int nDP2P1M3;
       int nDP2P1M4;
       int nDP2P2M1;
       int nDP2P2M2;
       int nDP2P2M3;
       int nDP3P1M1;
       int nDP3P2M1;
       int nDM1P1M1;
       int nDM1P1M2;
       int nDM1P1M3;
       int nDM1P1M4;
       int nDM1P2M1;
       int nDM1P2M2;
       int nDM1P2M3;
       int nDM2P1M1;
       int nDM2P1M2;
       int nDM2P1M3;
       int nDM2P1M4;
       int nDM2P2M1;
       int nDM2P2M2;
       int nDM2P2M3;
       int nDM3P1M1;
       int nDM3P2M1;
       int nL1M1;
       int nL1M2;
       int nL1M3;
       int nL1M4;
       int nL2M1;
       int nL2M2;
       int nL2M3;
       int nL2M4;
       int nL3M1;
       int nL3M2;
       int nL3M3;
       int nL3M4;
       int nL4M1;
       int nL4M2;
       int nL4M3;
       int nL4M4;
       int nBigEvents;
       int nBPIXDigis;
       int nFPIXDigis;
       MonitorElement* bigEventRate;
       MonitorElement* pixEvtsPerBX;
       MonitorElement* pixEventRate;
       MonitorElement* noOccROCsBarrel;
       MonitorElement* loOccROCsBarrel;
       MonitorElement* noOccROCsEndcap;
       MonitorElement* loOccROCsEndcap;
       MonitorElement* averageDigiOccupancy;
       MonitorElement* avgfedDigiOccvsLumi;
       MonitorElement* meNDigisCOMBBarrel_;
       MonitorElement* meNDigisCOMBEndcap_;
       MonitorElement* meNDigisCHANBarrel_;
       MonitorElement* meNDigisCHANBarrelL1_;
       MonitorElement* meNDigisCHANBarrelL2_;
       MonitorElement* meNDigisCHANBarrelL3_;
       MonitorElement* meNDigisCHANBarrelL4_;
       MonitorElement* meNDigisCHANBarrelCh1_;
       MonitorElement* meNDigisCHANBarrelCh2_;
       MonitorElement* meNDigisCHANBarrelCh3_;
       MonitorElement* meNDigisCHANBarrelCh4_;
       MonitorElement* meNDigisCHANBarrelCh5_;
       MonitorElement* meNDigisCHANBarrelCh6_;
       MonitorElement* meNDigisCHANBarrelCh7_;
       MonitorElement* meNDigisCHANBarrelCh8_;
       MonitorElement* meNDigisCHANBarrelCh9_;
       MonitorElement* meNDigisCHANBarrelCh10_;
       MonitorElement* meNDigisCHANBarrelCh11_;
       MonitorElement* meNDigisCHANBarrelCh12_;
       MonitorElement* meNDigisCHANBarrelCh13_;
       MonitorElement* meNDigisCHANBarrelCh14_;
       MonitorElement* meNDigisCHANBarrelCh15_;
       MonitorElement* meNDigisCHANBarrelCh16_;
       MonitorElement* meNDigisCHANBarrelCh17_;
       MonitorElement* meNDigisCHANBarrelCh18_;
       MonitorElement* meNDigisCHANBarrelCh19_;
       MonitorElement* meNDigisCHANBarrelCh20_;
       MonitorElement* meNDigisCHANBarrelCh21_;
       MonitorElement* meNDigisCHANBarrelCh22_;
       MonitorElement* meNDigisCHANBarrelCh23_;
       MonitorElement* meNDigisCHANBarrelCh24_;
       MonitorElement* meNDigisCHANBarrelCh25_;
       MonitorElement* meNDigisCHANBarrelCh26_;
       MonitorElement* meNDigisCHANBarrelCh27_;
       MonitorElement* meNDigisCHANBarrelCh28_;
       MonitorElement* meNDigisCHANBarrelCh29_;
       MonitorElement* meNDigisCHANBarrelCh30_;
       MonitorElement* meNDigisCHANBarrelCh31_;
       MonitorElement* meNDigisCHANBarrelCh32_;
       MonitorElement* meNDigisCHANBarrelCh33_;
       MonitorElement* meNDigisCHANBarrelCh34_;
       MonitorElement* meNDigisCHANBarrelCh35_;
       MonitorElement* meNDigisCHANBarrelCh36_;
       MonitorElement* meNDigisCHANEndcap_;
       MonitorElement* meNDigisCHANEndcapDp1_;
       MonitorElement* meNDigisCHANEndcapDp2_;
       MonitorElement* meNDigisCHANEndcapDp3_;
       MonitorElement* meNDigisCHANEndcapDm1_;
       MonitorElement* meNDigisCHANEndcapDm2_;
       MonitorElement* meNDigisCHANEndcapDm3_;
 };

#endif
