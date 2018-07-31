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
// $Id: CastorDumpConditions.cc,v 1.1 2011/05/09 19:38:47 mundim Exp $
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
      ~CastorDumpConditions() override;

       template<class S, class SRcd> void dumpIt(const std::vector<std::string>& mDumpRequest,
                                                 const edm::Event& e,
                                                 const edm::EventSetup& context,
                                                 const std::string name);

   private:
      std::string file_prefix;
      std::vector<std::string> mDumpRequest;
      virtual void beginJob(const edm::EventSetup&) ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;

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

   // dumpIt called for all possible ValueMaps. The function checks if the dump is actually requested.
   dumpIt<CastorElectronicsMap , CastorElectronicsMapRcd> (mDumpRequest, iEvent,iSetup,"ElectronicsMap" );
   dumpIt<CastorQIEData        , CastorQIEDataRcd>        (mDumpRequest, iEvent,iSetup,"ElectronicsMap" );
   dumpIt<CastorPedestals      , CastorPedestalsRcd>      (mDumpRequest, iEvent,iSetup,"Pedestals"      );
   dumpIt<CastorPedestalWidths , CastorPedestalWidthsRcd> (mDumpRequest, iEvent,iSetup,"PedestalWidths" );
   dumpIt<CastorGains          , CastorGainsRcd>          (mDumpRequest, iEvent,iSetup,"Gains"          );
   dumpIt<CastorGainWidths     , CastorGainWidthsRcd>     (mDumpRequest, iEvent,iSetup,"GainWidths"     );
   dumpIt<CastorChannelQuality , CastorChannelQualityRcd> (mDumpRequest, iEvent,iSetup,"ChannelQuality" );
   dumpIt<CastorRecoParams     , CastorRecoParamsRcd>     (mDumpRequest, iEvent,iSetup,"RecoParams"     );
   dumpIt<CastorSaturationCorrs, CastorSaturationCorrsRcd>(mDumpRequest, iEvent,iSetup,"SaturationCorrs");
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
void CastorDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name) {
    if (std::find (mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end())
    {
        int myrun = e.id().run();
        edm::ESHandle<S> p;
        context.get<SRcd>().get(p);
        S myobject(*p.product());

        std::ostringstream file;
        file << file_prefix << name.c_str() << "_Run" << myrun << ".txt";
        std::ofstream outStream(file.str().c_str() );
        CastorDbASCIIIO::dumpObject (outStream, myobject );
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorDumpConditions);
