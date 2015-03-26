// -*- C++ -*-
//
// Package:    MapTester
// Class:      MapTester
// 
/**\class MapTester MapTester.cc UserCode/MapTester/src/MapTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jared Todd Sturdy
//         Created:  Thu Oct 23 18:16:33 CEST 2008
// $Id: MapTester.cc,v 1.6 2011/11/15 14:19:38 pbgeff Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//
// class decleration
//

class MapTester : public edm::EDAnalyzer 
{
   public:
      explicit MapTester(const edm::ParameterSet&);
      ~MapTester();

   private:

      unsigned int mapIOV_;  //1 for first set, 2 for second, ...
      bool generateHTRLmap_;
      bool generateEmap_;
      bool generateOfflineDB_;
      bool generateQIEMap_;
      bool generateuHTRLmap_;

      virtual void beginRun(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

MapTester::MapTester(const edm::ParameterSet& iConfig)
{
  std::cout<<"test"<<std::endl;
  mapIOV_            = iConfig.getParameter<unsigned int>("mapIOV");
  generateHTRLmap_   = iConfig.getParameter<bool>("generateHTRLmap");
  generateuHTRLmap_  = iConfig.getParameter<bool>("generateuHTRLmap");
  generateEmap_      = iConfig.getParameter<bool>("generateEmap");
  generateOfflineDB_ = iConfig.getParameter<bool>("generateOfflineDB");
  generateQIEMap_    = iConfig.getParameter<bool>("generateQIEMap");
}


MapTester::~MapTester()
{

}

// ------------ method called to for each event  ------------
void MapTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout<<"test"<<std::endl;
  using namespace edm;
  beginRun(iSetup);
}


// ------------ method called once each job just before starting event loop  ------------
void MapTester::beginRun(const edm::EventSetup& iSetup)
{
  std::cout<<"test1"<<std::endl;
  char tempbuff[128];

  time_t myTime;
  time(&myTime);

  strftime(tempbuff,128,"%d.%b.%Y",localtime(&myTime) );

  edm::ESHandle<HcalTopology> topo;
  iSetup.get<IdealGeometryRecord>().get(topo);
  
  HcalLogicalMapGenerator mygen;
  HcalLogicalMap mymap = mygen.createMap(&(*topo),mapIOV_);

  //generate logical map for old HTR backend
  if (generateHTRLmap_) mymap.printHTRLMap(mapIOV_);
  //generate logical map for micro TCA backend, added by hua.wei@cern.ch
  if (generateuHTRLmap_) mymap.printuHTRLMap(mapIOV_);
  //generate QIE map, added by hua.wei@cern.ch
  if (generateQIEMap_) mymap.printQIEMap(mapIOV_);
  //generate Offline Database, added by hua.wei@cern.ch
  if (generateOfflineDB_) mymap.printOfflineDB(mapIOV_);
  

  mymap.checkIdFunctions();
  mymap.checkHashIds();
  mymap.checkElectronicsHashIds();

  if (generateEmap_)
  {
    std::ostringstream file;
    if      (mapIOV_==1) file<<"version_A_emap.txt";
    else if (mapIOV_==2) file<<"version_B_emap.txt";
    else if (mapIOV_==3) file<<"version_C_emap.txt";
    else if (mapIOV_==4) file<<"version_D_emap.txt";
    else if (mapIOV_==5) file<<"version_E_emap.txt";
    else if (mapIOV_==6) file<<"version_F_emap.txt";
    else                 file<<"version_G_emap.txt";

    std::ofstream outStream( file.str().c_str() );
    char buf [1024];
    sprintf(buf,"#file creation series : %s",tempbuff);
    outStream<<buf<< std::endl;

    HcalElectronicsMap myemap;
    edm::LogInfo("MapTester") <<"generating the emap..."<<std::endl;
    myemap = mymap.generateHcalElectronicsMap();
    edm::LogInfo("MapTester") <<"dumping the emap..."<<std::endl;
    HcalDbASCIIIO::dumpObject(outStream,myemap);

    //to output emap for uHTR, we do not use HcalElectronicsMap--dumpObject working chain; Instead, we get emap information directly from Lmap
    mymap.printuHTREMap(outStream);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void MapTester::endJob() 
{

}

//define this as a plug-in
DEFINE_FWK_MODULE(MapTester);
