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
using namespace dd4hep;

class DDTestDumpFile : public one::EDAnalyzer<> {
public:
  explicit DDTestDumpFile(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const string m_tag;
  const string m_outputFileName;
  const ESInputTag m_label;
};

DDTestDumpFile::DDTestDumpFile(const ParameterSet& iConfig)
    : m_tag(iConfig.getUntrackedParameter<string>("tag", "unknown")),
      m_outputFileName(iConfig.getUntrackedParameter<string>("outputFileName", "cmsDD4HepGeom.root")),
      m_label(iConfig.getParameter<ESInputTag>("DDDetector")) {}

void DDTestDumpFile::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestDumpFile::analyze: " << m_label;
  ESTransientHandle<DDDetector> det;
  iEventSetup.get<IdealGeometryRecord>().get(m_label, det);

  TGeoManager& geom = det->manager();

  int level = 1 + geom.GetTopVolume()->CountNodes(100, 3);

  LogVerbatim("Geometry") << "In the DDTestDumpFile::analyze method...obtained main geometry, level=" << level;

  TFile file(m_outputFileName.c_str(), "RECREATE");
  file.WriteTObject(&geom);
  file.WriteTObject(new TNamed("CMSSW_VERSION", gSystem->Getenv("CMSSW_VERSION")));
  file.WriteTObject(new TNamed("tag", m_tag.c_str()));
  file.Close();
}

DEFINE_FWK_MODULE(DDTestDumpFile);
