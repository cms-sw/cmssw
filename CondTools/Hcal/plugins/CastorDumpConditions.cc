// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorDumpConditions
// 
/**\class Castor CastorDumpConditions.cc CondTools/Castor/src/CastorDumpConditions.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luiz Mundim Filho
//         Created:  Thu Mar 12 14:45:44 CET 2009
// $Id: CastorDumpConditions.cc,v 1.2 2012/11/14 13:55:13 mundim Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
//
// class decleration
//

class CastorDumpConditions : public edm::EDAnalyzer {
   public:
      explicit CastorDumpConditions(const edm::ParameterSet&);
      ~CastorDumpConditions();

       template<class S, class SRcd> void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name);

   private:
      std::string file_prefix;
      std::vector<std::string> mDumpRequest;
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
CastorDumpConditions::CastorDumpConditions(const edm::ParameterSet& iConfig)

{
   file_prefix = iConfig.getUntrackedParameter<std::string>("outFilePrefix","Dump");
   mDumpRequest= iConfig.getUntrackedParameter<std::vector<std::string> >("dump",std::vector<std::string>());
   if (mDumpRequest.empty()) {
      std::cout << "CastorDumpConditions: No record to dump. Exiting." << std::endl;
      exit(0);
   }

}


CastorDumpConditions::~CastorDumpConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CastorDumpConditions::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
   std::cout << "I AM IN THE RUN " << iEvent.id().run() << std::endl;
   std::cout << "What to dump? "<< std::endl;
   if (mDumpRequest.empty()) {
      std::cout<< "CastorDumpConditions: Empty request" << std::endl;
      return;
   }

   for(std::vector<std::string>::const_iterator it=mDumpRequest.begin();it!=mDumpRequest.end();it++)
      std::cout << *it << std::endl;

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end())
      dumpIt(new CastorElectronicsMap(), new CastorElectronicsMapRcd(), iEvent,iSetup,"ElectronicsMap");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end())
      dumpIt(new CastorQIEData(), new CastorQIEDataRcd(), iEvent,iSetup,"QIEData"); 

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) 
      dumpIt(new CastorPedestals(), new CastorPedestalsRcd(), iEvent,iSetup,"Pedestals");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end())
      dumpIt(new CastorPedestalWidths(), new CastorPedestalWidthsRcd(), iEvent,iSetup,"PedestalWidths");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end())
      dumpIt(new CastorGains(), new CastorGainsRcd(), iEvent,iSetup,"Gains");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end())
      dumpIt(new CastorGainWidths(), new CastorGainWidthsRcd(), iEvent,iSetup,"GainWidths");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end())
      dumpIt(new CastorChannelQuality(), new CastorChannelQualityRcd(), iEvent,iSetup,"ChannelQuality");

    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RecoParams")) != mDumpRequest.end())
      dumpIt(new CastorRecoParams(), new CastorRecoParamsRcd(), iEvent,iSetup,"RecoParams");
      
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("SaturationCorrs")) != mDumpRequest.end())
      dumpIt(new CastorSaturationCorrs(), new CastorSaturationCorrsRcd(), iEvent,iSetup,"SaturationCorrs");

/*
   ESHandle<CastorPedestals> p;
   iSetup.get<CastorPedestalsRcd>().get(p);
   CastorPedestals* mypeds = new CastorPedestals(*p.product());
   std::ostringstream file;
   std::string name = "CastorPedestal";
   file << file_prefix << name.c_str() << "_Run" << iEvent.id().run()<< ".txt";
   std::ofstream outStream(file.str().c_str() );
   std::cout << "CastorDumpConditions: ---- Dumping " << name.c_str() << " ----" << std::endl;
   CastorDbASCIIIO::dumpObject (outStream, (*mypeds) );

*/   
}


// ------------ method called once each job just before starting event loop  ------------
void 
CastorDumpConditions::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CastorDumpConditions::endJob() {
}

template<class S, class SRcd>
  void CastorDumpConditions::dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name) {
    int myrun = e.id().run();
    edm::ESHandle<S> p;
    context.get<SRcd>().get(p);
    S* myobject = new S(*p.product());

    std::ostringstream file;
    file << file_prefix << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str() );
    CastorDbASCIIIO::dumpObject (outStream, (*myobject) );
  }

//define this as a plug-in
DEFINE_FWK_MODULE(CastorDumpConditions);
