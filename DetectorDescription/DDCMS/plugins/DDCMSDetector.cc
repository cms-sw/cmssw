#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>

class DDCMSDetector : public edm::one::EDAnalyzer<> {
public:
  explicit DDCMSDetector(const edm::ParameterSet& p);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  
  std::string confGeomXMLFiles_;
  std::vector< std::string > relFiles_;
  std::vector< std::string > files_;
};

DDCMSDetector::DDCMSDetector(const edm::ParameterSet& iConfig)
  : confGeomXMLFiles_(iConfig.getParameter<std::string>("confGeomXMLFiles"))
{
  relFiles_ = iConfig.getParameter<std::vector<std::string> >("geomXMLFiles");
  for( auto rit : relFiles_ ) {
    edm::FileInPath fp(rit);
    files_.emplace_back(fp.fullPath());
  }
}

void
DDCMSDetector::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& /*iSetup*/ )
{
  dd4hep::Detector& description = dd4hep::Detector::getInstance();

  std::string name("DD4hep_CompactLoader");
  const char* files[] = { confGeomXMLFiles_.c_str(), nullptr };
  description.apply(name.c_str(),2,(char**)files);
  for( auto it : files_ )
    std::cout << it << std::endl;
}

DEFINE_FWK_MODULE(DDCMSDetector);
