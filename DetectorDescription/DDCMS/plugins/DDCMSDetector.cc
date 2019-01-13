#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>

using namespace std;
using namespace cms;
using namespace dd4hep;

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

void
DDCMSDetector::analyze( const edm::Event&, const edm::EventSetup& iEventSetup)
{
  edm::ESTransientHandle<DDDetector> det;
  iEventSetup.get<DetectorDescriptionRcd>().get(det);

  edm::ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(registry);

  for( const auto& it : m_files )
    std::cout << it << std::endl;

  std::cout << "DD Vector Registry size: " << registry->vectors.size() << "\n";
  for( const auto& p: registry->vectors ) {
    std::cout << " " << p.first << " => ";
    for( const auto& i : p.second )
      std::cout << i << ", ";
    std::cout << '\n';
  }
  std::cout << "Iterate over the detectors:\n";
  for( auto const& it : det->description->detectors()) {
    dd4hep::DetElement det(it.second);
    std::cout << it.first << ": " << det.path() << "\n";
  }
  std::cout << "..done!\n";
}

void
DDCMSDetector::endJob()
{
}

DEFINE_FWK_MODULE( DDCMSDetector );
