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
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// system include files
#include <iostream>
#include <memory>

//
// class decleration
//

class TestPSetAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit TestPSetAnalyzer(edm::ParameterSet const&);

private:
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;

  edm::LuminosityBlockID testLumi_;
  edm::LuminosityBlockRange testLRange_;
  edm::EventRange testERange_;

  std::vector<edm::LuminosityBlockID> testVLumi_;
  std::vector<edm::LuminosityBlockRange> testVLRange_;
  std::vector<edm::EventRange> testVERange_;

  edm::EventID testEventID1_;
  edm::EventID testEventID2_;
  edm::EventID testEventID3_;
  edm::EventID testEventID4_;
  std::vector<edm::EventID> testVEventID_;
  edm::EventRange testERange1_;
  edm::EventRange testERange2_;
  edm::EventRange testERange3_;
  edm::EventRange testERange4_;
  edm::EventRange testERange5_;
  std::vector<edm::EventRange> testVERange2_;

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
  testLumi_ = iConfig.getParameter<edm::LuminosityBlockID>("testLumi");
  testVLumi_ = iConfig.getParameter<std::vector<edm::LuminosityBlockID> >("testVLumi");
  testLRange_ = iConfig.getParameter<edm::LuminosityBlockRange>("testRange");
  testVLRange_ = iConfig.getParameter<std::vector<edm::LuminosityBlockRange> >("testVRange");
  testERange_ = iConfig.getParameter<edm::EventRange>("testERange");
  testVERange_ = iConfig.getParameter<std::vector<edm::EventRange> >("testVERange");

  testEventID1_ = iConfig.getParameter<edm::EventID>("testEventID1");
  testEventID2_ = iConfig.getParameter<edm::EventID>("testEventID2");
  testEventID3_ = iConfig.getParameter<edm::EventID>("testEventID3");
  testEventID4_ = iConfig.getParameter<edm::EventID>("testEventID4");
  testVEventID_ = iConfig.getParameter<std::vector<edm::EventID> >("testVEventID");
  testERange1_ = iConfig.getParameter<edm::EventRange>("testERange1");
  testERange2_ = iConfig.getParameter<edm::EventRange>("testERange2");
  testERange3_ = iConfig.getParameter<edm::EventRange>("testERange3");
  testERange4_ = iConfig.getParameter<edm::EventRange>("testERange4");
  testERange5_ = iConfig.getParameter<edm::EventRange>("testERange5");
  testVERange2_ = iConfig.getParameter<std::vector<edm::EventRange> >("testVERange2");

  std::cout << "Lumi PSet test " << testLumi_ << std::endl;
  std::cout << "LRange PSet test " << testLRange_ << std::endl;
  std::cout << "ERange PSet test " << testERange_ << std::endl;

  for (auto const& i : testVLumi_) {
    std::cout << "VLumi PSet test " << i << std::endl;
  }

  for (auto const& i : testVLRange_) {
    std::cout << "VLRange PSet test " << i << std::endl;
  }

  for (auto const& i : testVERange_) {
    std::cout << "VERange PSet test " << i << std::endl;
  }

  std::cout << "EventID1 PSet test " << testEventID1_ << std::endl;
  std::cout << "EventID2 PSet test " << testEventID2_ << std::endl;
  std::cout << "EventID3 PSet test " << testEventID3_ << std::endl;
  std::cout << "EventID4 PSet test " << testEventID4_ << std::endl;
  std::cout << "ERange1 PSet test " << testERange1_ << std::endl;
  std::cout << "ERange2 PSet test " << testERange2_ << std::endl;
  std::cout << "ERange3 PSet test " << testERange3_ << std::endl;
  std::cout << "ERange4 PSet test " << testERange4_ << std::endl;
  std::cout << "ERange5 PSet test " << testERange5_ << std::endl;

  for (auto const& i : testVEventID_) {
    std::cout << "VEventID PSet test " << i << std::endl;
  }

  for (auto const& i : testVERange2_) {
    std::cout << "VERange2 PSet test " << i << std::endl;
  }
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestPSetAnalyzer::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {}

//define this as a plug-in
DEFINE_FWK_MODULE(TestPSetAnalyzer);
