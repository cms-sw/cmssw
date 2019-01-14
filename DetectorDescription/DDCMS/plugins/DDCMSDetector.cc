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
  string m_label;
};

DDCMSDetector::DDCMSDetector(const edm::ParameterSet& iConfig)
  : m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", ""))
{}

void
DDCMSDetector::analyze( const edm::Event&, const edm::EventSetup& iEventSetup)
{
  edm::ESTransientHandle<DDDetector> det;
  iEventSetup.get<DetectorDescriptionRcd>().get(m_label, det);

  cout << "Iterate over the detectors:\n";
  for( auto const& it : det->description->detectors()) {
    dd4hep::DetElement det(it.second);
    cout << it.first << ": " << det.path() << "\n";
  }
  cout << "..done!\n";
  
  edm::ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(m_label, registry);

  cout << "DD Vector Registry size: " << registry->vectors.size() << "\n";
  for( const auto& p: registry->vectors ) {
    cout << " " << p.first << " => ";
    for( const auto& i : p.second )
      cout << i << ", ";
    cout << '\n';
  }
}

void
DDCMSDetector::endJob()
{
}

DEFINE_FWK_MODULE( DDCMSDetector );
