// -*- C++ -*-
//
// Package:    DQMServices/Demo
// Class:      DemoNormalDQMEDAnalyzer
//
/**\class DemoNormalDQMEDAnalyzer DemoNormalDQMEDAnalyzer.cc DQMServices/Demo/plugins/DemoNormalDQMEDAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marcel Schneider
//         Created:  Wed, 22 May 2019 15:18:07 GMT
//
//

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DemoNormalDQMEDAnalyzer : public DQMEDAnalyzer {
public:
  explicit DemoNormalDQMEDAnalyzer(const edm::ParameterSet&);
  ~DemoNormalDQMEDAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string folder_;
  MonitorElement* example_;
  MonitorElement* example2D_;
  MonitorElement* example3D_;
  MonitorElement* exampleTProfile_;
  MonitorElement* exampleTProfile2D_;
};

DemoNormalDQMEDAnalyzer::DemoNormalDQMEDAnalyzer(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {
  // now do what ever initialization is needed
}

DemoNormalDQMEDAnalyzer::~DemoNormalDQMEDAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called for each event  ------------
void DemoNormalDQMEDAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  example_->Fill(5);
  example2D_->Fill(1.0, 2.0);
  example3D_->Fill(1.0, 2.0, 3.0);
  exampleTProfile_->Fill(1.0, 2.0);
  exampleTProfile2D_->Fill(1.0, 2.0, 3.0);
}

void DemoNormalDQMEDAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                             edm::Run const& run,
                                             edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  example_ = ibook.book1D(
      "EXAMPLE", "Example 1D", 20, 0., 10., [](TH1*) { edm::LogInfo("DemoNormalDQMEDAnalyzer") << "booked!\n"; });
  example2D_ = ibook.book2D("EXAMPLE_2D", "Example 2D", 20, 0, 20, 15, 0, 15);
  example3D_ = ibook.book3D("EXAMPLE_3D", "Example 3D", 20, 0, 20, 15, 0, 15, 25, 0, 25);
  exampleTProfile_ = ibook.bookProfile("EXAMPLE_TPROFILE", "Example TProfile", 20, 0, 20, 15, 0, 15);
  exampleTProfile2D_ = ibook.bookProfile2D("EXAMPLE_TPROFILE2D", "Example TProfile 2D", 20, 0, 20, 15, 0, 15, 0, 100);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoNormalDQMEDAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MY_FOLDER");
  descriptions.add("demo", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DemoNormalDQMEDAnalyzer);
