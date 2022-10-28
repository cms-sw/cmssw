#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"

#include <iostream>

class TrackerParametersGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerParametersGeometryAnalyzer(const edm::ParameterSet&)
      : parToken_(esConsumes()), geoToken_(esConsumes()) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> parToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geoToken_;
};

void TrackerParametersGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("TrackerParametersGeometryAnalyzer") << "Here I am";

  const auto& ptp = iSetup.getHandle(parToken_);
  const auto& pDD = iSetup.getHandle(geoToken_);

  GeometricDet const* gd = pDD->trackerDet();
  GeometricDet::ConstGeometricDetContainer subdetgd = gd->components();

  for (const auto& git : subdetgd) {
    edm::LogVerbatim("TrackerParametersGeometryAnalyzer") << git->name() << ": " << git->type();
  }
  for (const auto& vitem : ptp->vitems) {
    edm::LogVerbatim("TrackerParametersGeometryAnalyzer")
        << vitem.id << " is " << pDD->geomDetSubDetector(vitem.id) << " has " << vitem.vpars.size() << ":";
    for (const auto& in : vitem.vpars) {
      edm::LogVerbatim("TrackerParametersGeometryAnalyzer") << in << ";";
    }
  }
  for (int vpar : ptp->vpars) {
    edm::LogVerbatim("TrackerParametersGeometryAnalyzer") << vpar << "; ";
  }
}

DEFINE_FWK_MODULE(TrackerParametersGeometryAnalyzer);
