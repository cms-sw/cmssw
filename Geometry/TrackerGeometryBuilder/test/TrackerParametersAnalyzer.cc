#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

#include <iostream>

class TrackerParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerParametersAnalyzer(const edm::ParameterSet&) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void TrackerParametersAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("TrackerParametersAnalyzer") << "Here I am";

  edm::ESHandle<PTrackerParameters> ptp;
  iSetup.get<PTrackerParametersRcd>().get(ptp);

  for (const auto& vitem : ptp->vitems) {
    edm::LogVerbatim("TrackerParametersAnalyzer") << vitem.id << " has " << vitem.vpars.size() << ":";
    for (const auto& in : vitem.vpars) {
      edm::LogVerbatim("TrackerParametersAnalyzer") << in << ";";
    }
  }
  for (int vpar : ptp->vpars) {
    std::cout << vpar << "; ";
  }
}

DEFINE_FWK_MODULE(TrackerParametersAnalyzer);
