#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"

#include <iostream>

using namespace std;
using namespace cms;

class DDTestVectors : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestVectors(const edm::ParameterSet& ) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void
DDTestVectors::analyze( const edm::Event&, const edm::EventSetup& iEventSetup)
{
  std::cout << "DDTestVectors::analyze:\n";
  edm::ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(registry);

  std::cout << "DD Vector Registry size: " << registry->vectors.size() << "\n";
  for( const auto& p: registry->vectors ) {
    std::cout << " " << p.first << " => ";
    for( const auto& i : p.second )
      std::cout << i << ", ";
    std::cout << '\n';
  }
}

DEFINE_FWK_MODULE(DDTestVectors);
