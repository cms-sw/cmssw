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

class DDTestDumpGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestDumpGeometry(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

DDTestDumpGeometry::DDTestDumpGeometry(const edm::ParameterSet& iConfig)
{}

void
DDTestDumpGeometry::analyze(const edm::Event&, const edm::EventSetup& iEventSetup)
{
  cout << "DDTestDumpGeometry::analyze:\n";
  edm::ESTransientHandle<DDDetector> det;
  iEventSetup.get<DetectorDescriptionRcd>().get(det);

  TGeoManager& geom = det->description->manager();

  TGeoIterator next(geom.GetTopVolume());
  TGeoNode *node;
  TString path;
  while(( node = next())) {
    next.GetPath( path );
    cout << path << ": "<< node->GetVolume()->GetName() << "\n";
  }
}

DEFINE_FWK_MODULE( DDTestDumpGeometry );
