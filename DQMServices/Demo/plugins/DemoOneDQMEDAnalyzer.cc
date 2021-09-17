// -*- C++ -*-
//
// Package:    DQMServices/Demo
// Class:      DemoOneDQMEDAnalyzer
//
/**\class DemoOneDQMEDAnalyzer DemoOneDQMEDAnalyzer.cc DQMServices/Demo/plugins/DemoOneDQMEDAnalyzer.cc

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
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DemoOneDQMEDAnalyzer : public DQMOneEDAnalyzer<> {
public:
  explicit DemoOneDQMEDAnalyzer(const edm::ParameterSet&);
  ~DemoOneDQMEDAnalyzer() override;

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
  int eventCount_ = 0;
};

DemoOneDQMEDAnalyzer::DemoOneDQMEDAnalyzer(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {
  // now do what ever initialization is needed
}

DemoOneDQMEDAnalyzer::~DemoOneDQMEDAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called for each event  ------------
void DemoOneDQMEDAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  eventCount_++;

  example_->Fill(5);
  example2D_->Fill(eventCount_ / 10, eventCount_ / 10);
  example3D_->Fill(eventCount_ / 10, eventCount_ / 10, eventCount_ / 10.f);
  exampleTProfile_->Fill(eventCount_ / 10, eventCount_ / 10.f);
  exampleTProfile2D_->Fill(eventCount_ / 10, eventCount_ / 10, eventCount_ / 10.f);
}

void DemoOneDQMEDAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  example_ = ibook.book1D("EXAMPLE", "Example 1D", 20, 0., 10.);
  example2D_ = ibook.book2D("EXAMPLE_2D", "Example 2D", 20, 0, 20, 15, 0, 15);
  example3D_ = ibook.book3D("EXAMPLE_3D", "Example 3D", 20, 0, 20, 15, 0, 15, 25, 0, 25);
  exampleTProfile_ = ibook.bookProfile("EXAMPLE_TPROFILE", "Example TProfile", 20, 0, 20, 15, 0, 15);
  exampleTProfile2D_ = ibook.bookProfile2D("EXAMPLE_TPROFILE2D", "Example TProfile 2D", 20, 0, 20, 15, 0, 15, 0, 100);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoOneDQMEDAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MY_FOLDER");
  descriptions.add("demoone", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DemoOneDQMEDAnalyzer);
