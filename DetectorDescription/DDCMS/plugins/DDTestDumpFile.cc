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

class DDTestDumpFile : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestDumpFile( const edm::ParameterSet& );

  void beginJob() override {}
  void analyze( edm::Event const& iEvent, edm::EventSetup const& ) override;
  void endJob() override {}

private:
  std::string m_tag;
  std::string m_outputFileName;
};

DDTestDumpFile::DDTestDumpFile( const edm::ParameterSet& iConfig )
{
  m_tag =  iConfig.getUntrackedParameter<std::string>( "tag", "unknown" );
  m_outputFileName = iConfig.getUntrackedParameter<std::string>( "outputFileName", "cmsDD4HepGeom.root" );
}

void
DDTestDumpFile::analyze( const edm::Event&, const edm::EventSetup& )
{
  std::cout << "DDTestDumpFile::analyze:\n";
  dd4hep::Detector& description = dd4hep::Detector::getInstance();
  TGeoManager& geom = description.manager();
  
  int level = 1 + geom.GetTopVolume()->CountNodes( 100, 3 );
  
  std::cout << "In the DDTestDumpFile::analyze method...obtained main geometry, level="
  	    << level << std::endl;

  TFile file( m_outputFileName.c_str(), "RECREATE" );
  file.WriteTObject( &geom );
  file.WriteTObject( new TNamed( "CMSSW_VERSION", gSystem->Getenv( "CMSSW_VERSION" )));
  file.WriteTObject( new TNamed( "tag", m_tag.c_str()));
  file.Close();
}

DEFINE_FWK_MODULE( DDTestDumpFile );
