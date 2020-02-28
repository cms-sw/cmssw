#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;

class DTGeometryTest : public one::EDAnalyzer<> {
public:
  explicit DTGeometryTest(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const string m_label;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> m_token;
};

DTGeometryTest::DTGeometryTest(const ParameterSet& iConfig)
    : m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", "")),
      m_token(esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{"", m_label})) {}

void DTGeometryTest::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("DTGeometryTest") << "DTGeometryTest::analyze: " << m_label;
  ESTransientHandle<DTGeometry> pDD = iEventSetup.getTransientHandle(m_token);

  LogVerbatim("DTGeometryTest") << " Geometry node for DTGeom is " << (pDD.isValid() ? "valid" : "not valid");
  LogVerbatim("DTGeometryTest") << " I have " << pDD->detTypes().size() << " detTypes";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->detUnits().size() << " detUnits";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->dets().size() << " dets";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->layers().size() << " layers";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->superLayers().size() << " superlayers";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->chambers().size() << " chambers";

  // check chamber
  LogVerbatim("DTGeometryTest") << "CHAMBERS " << string(120, '-');

  LogVerbatim("DTGeometryTest").log([&](auto& log) {
    for (auto det : pDD->chambers()) {
      const BoundPlane& surf = det->surface();
      log << "Chamber " << det->id() << " Position " << surf.position() << " normVect " << surf.normalVector()
          << " bounds W/H/L: " << surf.bounds().width() << "/" << surf.bounds().thickness() << "/"
          << surf.bounds().length() << "\n";
    }
  });
  LogVerbatim("DTGeometryTest") << "END " << string(120, '-');

  // check superlayers
  LogVerbatim("DTGeometryTest") << "SUPERLAYERS " << string(120, '-');
  LogVerbatim("DTGeometryTest").log([&](auto& log) {
    for (auto det : pDD->superLayers()) {
      const BoundPlane& surf = det->surface();
      log << "SuperLayer " << det->id() << " chamber " << det->chamber()->id() << " Position " << surf.position()
          << " normVect " << surf.normalVector() << " bounds W/H/L: " << surf.bounds().width() << "/"
          << surf.bounds().thickness() << "/" << surf.bounds().length() << "\n";
    }
  });
  LogVerbatim("DTGeometryTest") << "END " << string(120, '-');

  // check layers
  LogVerbatim("DTGeometryTest") << "LAYERS " << string(120, '-');

  LogVerbatim("DTGeometryTest").log([&](auto& log) {
    for (auto det : pDD->layers()) {
      const DTTopology& topo = det->specificTopology();
      const BoundPlane& surf = det->surface();
      log << "Layer " << det->id() << " SL " << det->superLayer()->id() << " chamber " << det->chamber()->id()
          << " Topology W/H/L: " << topo.cellWidth() << "/" << topo.cellHeight() << "/" << topo.cellLenght()
          << " first/last/# wire " << topo.firstChannel() << "/" << topo.lastChannel() << "/" << topo.channels()
          << " Position " << surf.position() << " normVect " << surf.normalVector()
          << " bounds W/H/L: " << surf.bounds().width() << "/" << surf.bounds().thickness() << "/"
          << surf.bounds().length() << "\n";
    }
  });
  LogVerbatim("DTGeometryTest") << "END " << string(120, '-');
}

DEFINE_FWK_MODULE(DTGeometryTest);
