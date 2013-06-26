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
// $Id: MapTester.cc,v 1.7 2012/04/04 09:32:27 abdullin Exp $
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
//
// class decleration
//

class MapTester : public edm::EDAnalyzer {
   public:
      explicit MapTester(const edm::ParameterSet&);
      ~MapTester();


   private:

      unsigned int mapIOV_;  //1 for first set, 2 for second, ...
      bool generateTextfiles_;
      bool generateEmap_;

      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

MapTester::MapTester(const edm::ParameterSet& iConfig)
{
  mapIOV_            = iConfig.getParameter<unsigned int>("mapIOV");
  generateTextfiles_ = iConfig.getParameter<bool>("generateTextfiles");
  generateEmap_      = iConfig.getParameter<bool>("generateEmap");
}


MapTester::~MapTester()
{

}

// ------------ method called to for each event  ------------
void
MapTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
}


// ------------ method called once each job just before starting event loop  ------------
void 
MapTester::beginJob()
{
  char tempbuff[128];

  time_t myTime;
  time(&myTime);

  strftime(tempbuff,128,"%d.%b.%Y",localtime(&myTime) );

  HcalLogicalMapGenerator mygen;
  HcalLogicalMap mymap=mygen.createMap(mapIOV_);

  if (generateTextfiles_) mymap.printMap(mapIOV_);

  mymap.checkIdFunctions();
  mymap.checkHashIds();
  mymap.checkElectronicsHashIds();

  if (generateEmap_){
    std::ostringstream file;
    if      (mapIOV_==1) file<<"version_A_emap.txt";
    else if (mapIOV_==2) file<<"version_B_emap.txt";
    else if (mapIOV_==3) file<<"version_C_emap.txt";
    else                 file<<"version_D_emap.txt";

    std::ofstream outStream( file.str().c_str() );
    char buf [1024];
    sprintf(buf,"#file creation series : %s",tempbuff);
    outStream<<buf<< std::endl;

    HcalElectronicsMap myemap;
    edm::LogInfo( "MapTester") <<"generating the emap..."<<std::endl;
    myemap = mymap.generateHcalElectronicsMap();
    edm::LogInfo( "MapTester") <<"dumping the emap..."<<std::endl;
    HcalDbASCIIIO::dumpObject(outStream,myemap);}
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MapTester::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MapTester);
