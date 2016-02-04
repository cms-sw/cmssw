// Test reading of values from the config file into
// an analyzer. Especially useful for some of the more
// complex data types

//
// Original Author:  Eric Vaandering
//         Created:  Mon Dec 22 13:43:10 CST 2008
//
//


// user include files
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// system include files
#include <memory>

//
// class decleration
//

class TestPSetAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestPSetAnalyzer(edm::ParameterSet const&);
      ~TestPSetAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(edm::Event const&, edm::EventSetup const&);
      virtual void endJob() ;

      edm::LuminosityBlockID                 testLumi_;
      edm::LuminosityBlockRange              testLRange_;
      edm::EventRange                        testERange_;

      std::vector<edm::LuminosityBlockID>    testVLumi_;
      std::vector<edm::LuminosityBlockRange> testVLRange_;
      std::vector<edm::EventRange>           testVERange_;

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
TestPSetAnalyzer::TestPSetAnalyzer(edm::ParameterSet const& iConfig) {

    testLumi_             = iConfig.getParameter<edm::LuminosityBlockID>("testLumi");
    testVLumi_            = iConfig.getParameter<std::vector<edm::LuminosityBlockID> >("testVLumi");
    testLRange_           = iConfig.getParameter<edm::LuminosityBlockRange>("testRange");
    testVLRange_          = iConfig.getParameter<std::vector<edm::LuminosityBlockRange> >("testVRange");
    testERange_           = iConfig.getParameter<edm::EventRange>("testERange");
    testVERange_          = iConfig.getParameter<std::vector<edm::EventRange> >("testVERange");

    std::cout << "Lumi PSet test "   << testLumi_  << std::endl;
    std::cout << "LRange PSet test "  << testLRange_ << std::endl;
    std::cout << "ERange PSet test "  << testERange_ << std::endl;

    for(std::vector<edm::LuminosityBlockID>::const_iterator i = testVLumi_.begin();
        i !=  testVLumi_.end(); ++i) {
      std::cout << "VLumi PSet test " << *i << std::endl;
    }

    for(std::vector<edm::LuminosityBlockRange>::const_iterator i = testVLRange_.begin();
        i !=  testVLRange_.end(); ++i) {
      std::cout << "VLRange PSet test " << *i << std::endl;
    }

    for(std::vector<edm::EventRange>::const_iterator i = testVERange_.begin();
        i !=  testVERange_.end(); ++i) {
      std::cout << "VERange PSet test " << *i << std::endl;
    }

   //now do what ever initialization is needed

}

TestPSetAnalyzer::~TestPSetAnalyzer() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
TestPSetAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {
   using namespace edm;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example", pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

// ------------ method called once each job just before starting event loop  ------------
void
TestPSetAnalyzer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void
TestPSetAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestPSetAnalyzer);
