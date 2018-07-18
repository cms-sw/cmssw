#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/DDCMS/interface/DDRegistry.h"

#include <iostream>

class DDTestVectors : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestVectors(const edm::ParameterSet& ) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void
DDTestVectors::analyze( const edm::Event&, const edm::EventSetup& )
{
  std::cout << "DDTestVectors::analyze:\n";
  DDVectorRegistry registry;

  std::cout << "DD Vector Registry size: " << registry->size() << "\n";
  for( const auto& p: *registry ) {
    std::cout << " " << p.first << " => ";
    for( const auto& i : p.second )
      std::cout << i << ", ";
    std::cout << '\n';
  }
}

DEFINE_FWK_MODULE(DDTestVectors);
