// -*- C++ -*-
//
// Package:    DQMServices/Demo
// Class:      DemoGlobalDQMEDAnalyzer
//
/**\class DemoGlobalDQMEDAnalyzer DemoGlobalDQMEDAnalyzer.cc DQMServices/Demo/plugins/DemoGlobalDQMEDAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marcel Schneider
//         Created:  Wed, 22 May 2019 15:18:23 GMT
//
//

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct Histograms_Demo2 {
  typedef dqm::reco::MonitorElement MonitorElement;
  MonitorElement* histo_;
};

class DemoGlobalDQMEDAnalyzer : public DQMGlobalEDAnalyzer<Histograms_Demo2> {
public:
  explicit DemoGlobalDQMEDAnalyzer(const edm::ParameterSet&);
  ~DemoGlobalDQMEDAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms_Demo2&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_Demo2 const&) const override;

  std::string folder_;
};

DemoGlobalDQMEDAnalyzer::DemoGlobalDQMEDAnalyzer(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {
  // now do what ever initialization is needed
}

DemoGlobalDQMEDAnalyzer::~DemoGlobalDQMEDAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called for each event  ------------
void DemoGlobalDQMEDAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                         edm::EventSetup const& iSetup,
                                         Histograms_Demo2 const& histos) const {
  histos.histo_->Fill(5);
}

void DemoGlobalDQMEDAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                             edm::Run const& run,
                                             edm::EventSetup const& iSetup,
                                             Histograms_Demo2& histos) const {
  ibook.setCurrentFolder(folder_);
  histos.histo_ = ibook.book1D("EXAMPLE", "EXAMPLE", 10, 0., 10.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoGlobalDQMEDAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MY_FOLDER");
  descriptions.add("demo2", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DemoGlobalDQMEDAnalyzer);
