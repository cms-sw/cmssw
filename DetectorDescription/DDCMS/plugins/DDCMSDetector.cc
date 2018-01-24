#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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
  
  std::string geomXMLFiles_;
};

DDCMSDetector::DDCMSDetector(const edm::ParameterSet& iConfig)
  : geomXMLFiles_(iConfig.getParameter<std::string>("geomXMLFiles"))
{}

void
DDCMSDetector::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& /*iSetup*/ )
{
  dd4hep::Detector& description = dd4hep::Detector::getInstance();

  std::string name("DD4hepCompactLoader");
  const char* files[] = { geomXMLFiles_.c_str(), nullptr };
  description.apply(name.c_str(),2,(char**)files);
}

DEFINE_FWK_MODULE(DDCMSDetector);
