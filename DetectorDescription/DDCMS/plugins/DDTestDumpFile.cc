#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
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
  string m_tag;
  string m_outputFileName;
  string m_label;
};

DDTestDumpFile::DDTestDumpFile(const ParameterSet& iConfig)
  : m_tag(iConfig.getUntrackedParameter<string>("tag", "unknown")),
    m_outputFileName(iConfig.getUntrackedParameter<string>("outputFileName", "cmsDD4HepGeom.root")),
    m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", ""))
{}

void
DDTestDumpFile::analyze(const Event&, const EventSetup& iEventSetup)
{
  cout << "DDTestDumpFile::analyze: " << m_label << "\n";
  ESTransientHandle<DDDetector> det;
  iEventSetup.get<DetectorDescriptionRcd>().get(m_label, det);

  TGeoManager& geom = det->description->manager();
  
  int level = 1 + geom.GetTopVolume()->CountNodes( 100, 3 );
  
  cout << "In the DDTestDumpFile::analyze method...obtained main geometry, level="
       << level << "\n";

  TFile file(m_outputFileName.c_str(), "RECREATE");
  file.WriteTObject(&geom );
  file.WriteTObject(new TNamed("CMSSW_VERSION", gSystem->Getenv("CMSSW_VERSION")));
  file.WriteTObject(new TNamed("tag", m_tag.c_str()));
  file.Close();
}

DEFINE_FWK_MODULE(DDTestDumpFile);
