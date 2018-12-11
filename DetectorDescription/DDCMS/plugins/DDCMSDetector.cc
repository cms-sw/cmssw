#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/DDCMS/interface/DDRegistry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>

class DDCMSDetector : public edm::one::EDAnalyzer<> {
public:
  explicit DDCMSDetector(const edm::ParameterSet& p);

  void beginJob() override {}
  void analyze( edm::Event const& iEvent, edm::EventSetup const& ) override;
  void endJob() override;

private:
  
  std::string m_confGeomXMLFiles;
  std::vector< std::string > m_relFiles;
  std::vector< std::string > m_files;
};

DDCMSDetector::DDCMSDetector( const edm::ParameterSet& iConfig )
{
  m_confGeomXMLFiles = edm::FileInPath( iConfig.getParameter<std::string>( "confGeomXMLFiles" )).fullPath();
  
  m_relFiles = iConfig.getParameter<std::vector<std::string> >( "geomXMLFiles" );
  for( const auto& it : m_relFiles ) {
    edm::FileInPath fp( it );
    m_files.emplace_back( fp.fullPath());
  }
}

#include "DD4hep/DetElement.h"

void
DDCMSDetector::analyze( const edm::Event&, const edm::EventSetup& )
{
  dd4hep::Detector& description = dd4hep::Detector::getInstance();

  std::string name( "DD4hep_CompactLoader" );

  const char* files[] = { m_confGeomXMLFiles.c_str(), nullptr };
  description.apply( name.c_str(), 2, (char**)files );

  for( const auto& it : m_files )
    std::cout << it << std::endl;

  DDVectorRegistry registry;
  std::cout << "DD Vector Registry size: " << registry->size() << "\n";
  for( const auto& p: *registry ) {
    std::cout << " " << p.first << " => ";
    for( const auto& i : p.second )
      std::cout << i << ", ";
    std::cout << '\n';
  }
  std::cout << "Iterate over the detectors:\n";
  for( dd4hep::Detector::HandleMap::const_iterator i = description.detectors().begin(); i != description.detectors().end(); ++i ) {
    dd4hep::DetElement det( (*i).second );
    std::cout << (*i).first << ": " << det.path() << "\n";
  }
  std::cout << "..done!\n";
}

void
DDCMSDetector::endJob()
{
  // FIXME: It does not clean up:
  // dd4hep::Detector::getInstance().destroyInstance();
}

DEFINE_FWK_MODULE( DDCMSDetector );
