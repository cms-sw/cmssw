#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DD4hep/Detector.h"

#include "TGeoManager.h"
#include "TFile.h"
#include "TSystem.h"

#include <memory>
#include <string>

class DDCMSDetector : public edm::one::EDAnalyzer<> {
public:
  explicit DDCMSDetector(const edm::ParameterSet& p);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  
  std::string m_confGeomXMLFiles;
  std::vector< std::string > m_relFiles;
  std::vector< std::string > m_files;
  std::string m_tag;
  std::string m_outputFileName;
};

DDCMSDetector::DDCMSDetector(const edm::ParameterSet& iConfig)
  : m_confGeomXMLFiles(iConfig.getParameter<std::string>("confGeomXMLFiles"))
{
  m_tag =  iConfig.getUntrackedParameter<std::string>("tag", "unknown");
  m_outputFileName = iConfig.getUntrackedParameter<std::string>("outputFileName", "cmsSimGeom.root");
  m_relFiles = iConfig.getParameter<std::vector<std::string> >("geomXMLFiles");
  for( auto rit : m_relFiles ) {
    edm::FileInPath fp(rit);
    m_files.emplace_back(fp.fullPath());
  }
}

void
DDCMSDetector::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& /*iSetup*/ )
{
  dd4hep::Detector& description = dd4hep::Detector::getInstance();

  std::string name("DD4hep_CompactLoader");
  
  const char* files[] = { m_confGeomXMLFiles.c_str(), nullptr };
  description.apply( name.c_str(), 2, (char**)files );
  for( auto it : m_files )
    std::cout << it << std::endl;

  TGeoManager& geom = description.manager();
  
  int level = 1 + geom.GetTopVolume()->CountNodes(100, 3);
  
  std::cout << "In the DumpSimGeometry::analyze method...obtained main geometry, level="
	    << level << std::endl;

  TFile f(m_outputFileName.c_str(), "RECREATE");
  f.WriteTObject(&geom);
  f.WriteTObject(new TNamed("CMSSW_VERSION", gSystem->Getenv( "CMSSW_VERSION" )));
  f.WriteTObject(new TNamed("tag", m_tag.c_str()));
  f.Close();
}

DEFINE_FWK_MODULE(DDCMSDetector);
