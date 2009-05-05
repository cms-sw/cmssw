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
// $Id: MapTester.cc,v 1.1 2009/01/24 15:09:24 rofierzy Exp $
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
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MapTester::MapTester(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


MapTester::~MapTester()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MapTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
MapTester::beginJob(const edm::EventSetup&)
{
  HcalLogicalMapGenerator mygen;
  HcalLogicalMap mymap=mygen.createMap();
  mymap.printMap();
  mymap.checkIdFunctions();
  mymap.checkHashIds();
  mymap.checkElectronicsHashIds();

  std::ostringstream file;
  file<<"myemaptest.txt";
  std::ofstream outStream( file.str().c_str() );

  HcalElectronicsMap myemap;
  std::cout<<"generating the emap..."<<std::endl;
  myemap = mymap.generateHcalElectronicsMap();
  std::cout<<"dumping the emap..."<<std::endl;
  HcalDbASCIIIO::dumpObject(outStream,myemap);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MapTester::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MapTester);
