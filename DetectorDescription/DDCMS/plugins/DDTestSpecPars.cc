#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestSpecPars : public one::EDAnalyzer<> {
public:
  explicit DDTestSpecPars(const ParameterSet& iConfig)
    : m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", "")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:  
  string m_label;
};

void
DDTestSpecPars::analyze(const Event&, const EventSetup& iEventSetup)
{
  cout << "DDTestSpecPars::analyze: " << m_label << "\n";
  ESTransientHandle<DDSpecParRegistry> registry;
  iEventSetup.get<DDSpecParRegistryRcd>().get(m_label, registry);

  cout << "DD SpecPar Registry size: " << registry->specpars.size() << "\n";
  for(const auto& i: registry->specpars) {
    cout << " " << i.first << " => ";
    for(const auto& k : i.second.paths)
      cout << k << ", ";
    for(const auto& l : i.second.spars) {
      cout << l.first << " => ";
      for(const auto& il : l.second) {
	cout << il << ", ";
      }
    }
    for(const auto& m : i.second.numpars) {
      cout << m.first << " => ";
      for(const auto& im : m.second) {
	cout << im << ", ";
      }
    }
    cout << '\n';
  }
}

DEFINE_FWK_MODULE(DDTestSpecPars);
