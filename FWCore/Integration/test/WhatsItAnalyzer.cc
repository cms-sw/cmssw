// -*- C++ -*-
//
// Package:    WhatsItAnalyzer
// Class:      WhatsItAnalyzer
// 
/**\class WhatsItAnalyzer WhatsItAnalyzer.cc test/WhatsItAnalyzer/src/WhatsItAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
//
//


// system include files
#include <memory>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/GadgetRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// class decleration
//

namespace edmtest {

class WhatsItAnalyzer : public edm::EDAnalyzer {
   public:
      explicit WhatsItAnalyzer(const edm::ParameterSet&);
      ~WhatsItAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:

      void getAndTest(GadgetRcd const& record,
                      edm::ESHandle<WhatsIt>& handle,
                      int expectedValue,
                      const char* label);

      // ----------member data ---------------------------
      std::vector<int> expectedValues_;
      unsigned int index_;
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
WhatsItAnalyzer::WhatsItAnalyzer(const edm::ParameterSet& iConfig):
   expectedValues_(iConfig.getUntrackedParameter<std::vector<int> >("expectedValues",std::vector<int>())),
   index_(0)
{
   //now do what ever initialization is needed

}


WhatsItAnalyzer::~WhatsItAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
WhatsItAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   edm::ESHandle<WhatsIt> pSetup;
   GadgetRcd const& gadgetRcd = iSetup.get<GadgetRcd>();

   if (index_ < expectedValues_.size()) {
      int expectedValue = expectedValues_.at(index_);
      getAndTest(gadgetRcd, pSetup, expectedValue, "");
      getAndTest(gadgetRcd, pSetup, expectedValue, "A");
      getAndTest(gadgetRcd, pSetup, expectedValue, "B");
      getAndTest(gadgetRcd, pSetup, expectedValue, "C");
      getAndTest(gadgetRcd, pSetup, expectedValue, "D");
      ++index_;
   }
}

void WhatsItAnalyzer::getAndTest(GadgetRcd const& record,
                                 edm::ESHandle<WhatsIt>& handle,
                                 int expectedValue,
                                 const char* label) {
   record.get(label, handle);
   if (expectedValue != handle->a) {
      throw cms::Exception("TestFail") << label << ": expected value " << expectedValue
         << " but got "<< handle->a;
   }
}
}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItAnalyzer);
