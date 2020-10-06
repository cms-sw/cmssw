#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include <DD4hep/SpecParRegistry.h>

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestSpecPars : public one::EDAnalyzer<> {
public:
  explicit DDTestSpecPars(const ParameterSet& iConfig) : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
};

void DDTestSpecPars::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestSpecPars::analyze: " << m_tag;
  ESTransientHandle<dd4hep::SpecParRegistry> registry;
  iEventSetup.get<DDSpecParRegistryRcd>().get(m_tag, registry);

  LogVerbatim("Geometry").log([&registry](auto& log) {
    log << "DD SpecPar Registry size: " << registry->specpars.size();
    for (const auto& i : registry->specpars) {
      log << " " << i.first << " == " << std::string({i.second.name.data(), i.second.name.size()}) << " =>";
      log << "\npaths:\n";
      for (const auto& k : i.second.paths)
        log << k << ", ";
      log << "\nstring parameters:\n";
      for (const auto& l : i.second.spars) {
        log << l.first << " == " << i.second.strValue(l.first) << " = ";
        for (const auto& il : l.second) {
          log << il << ", ";
        }
      }
      log << "\nnumeric parameters\n";
      for (const auto& m : i.second.numpars) {
        log << m.first << " => ";
        for (const auto& im : m.second) {
          log << im << ", ";
        }
      }
      log << '\n';
    }
  });
}

DEFINE_FWK_MODULE(DDTestSpecPars);
