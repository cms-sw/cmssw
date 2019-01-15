#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestVectors : public one::EDAnalyzer<> {
public:
  explicit DDTestVectors(const ParameterSet& iConfig)
    : m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", ""))
  {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:  
  string m_label;
};

void
DDTestVectors::analyze( const Event&, const EventSetup& iEventSetup)
{
  cout << "DDTestVectors::analyze: " << m_label << "\n";
  ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(m_label, registry);

  cout << "DD Vector Registry size: " << registry->vectors.size() << "\n";
  for(const auto& p: registry->vectors) {
    cout << " " << p.first << " => ";
    for(const auto& i : p.second)
      cout << i << ", ";
    cout << '\n';
  }
}

DEFINE_FWK_MODULE(DDTestVectors);
