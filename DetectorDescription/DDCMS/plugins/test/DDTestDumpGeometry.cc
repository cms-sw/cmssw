#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepRootPersistency.h"

#include "TGeoManager.h"
#include "TFile.h"
#include "TSystem.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestDumpGeometry : public one::EDAnalyzer<> {
public:
  explicit DDTestDumpGeometry(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
  const ESGetToken<DDDetector, IdealGeometryRecord> m_token;
};

DDTestDumpGeometry::DDTestDumpGeometry(const ParameterSet& iConfig)
    : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")), m_token(esConsumes(m_tag)) {}

void DDTestDumpGeometry::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestDumpGeometry::analyze: " << m_tag;
  ESTransientHandle<DDDetector> det = iEventSetup.getTransientHandle(m_token);

  TGeoManager const& geom = det->manager();

  TGeoIterator next(geom.GetTopVolume());
  TGeoNode* node;
  TString path;
  while ((node = next())) {
    next.GetPath(path);
    path.ReplaceAll("xml-memory-buffer:", "");  // Remove artifact of DB reading
    TString nodeName(node->GetVolume()->GetName());
    nodeName.ReplaceAll("xml-memory-buffer:", "");  // Remove artifact of DB reading
    LogVerbatim("DumpGeometry") << path << ": " << nodeName;
  }
}

DEFINE_FWK_MODULE(DDTestDumpGeometry);
