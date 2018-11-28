#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/DDCMS/interface/DDRegistry.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepRootPersistency.h"

#include "TGeoManager.h"
#include "TFile.h"
#include "TSystem.h"

#include <iostream>
#include <string>

class DDTestDumpGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestDumpGeometry( const edm::ParameterSet& );

  void beginJob() override {}
  void analyze( edm::Event const& iEvent, edm::EventSetup const& ) override;
  void endJob() override {}
};

DDTestDumpGeometry::DDTestDumpGeometry( const edm::ParameterSet& iConfig )
{}

void
DDTestDumpGeometry::analyze( const edm::Event&, const edm::EventSetup& )
{
  std::cout << "DDTestDumpGeometry::analyze:\n";
  dd4hep::Detector& description = dd4hep::Detector::getInstance();
  
  TGeoManager& geom = description.manager();

  TGeoIterator next( geom.GetTopVolume());
  TGeoNode *node;
  TString path;
  while(( node = next())) {
    next.GetPath( path );
    std::cout << path << ": "<< node->GetVolume()->GetName() << "\n";
  }
}

DEFINE_FWK_MODULE( DDTestDumpGeometry );
