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

class DDTestSpecParsFilter : public one::EDAnalyzer<> {
public:
  explicit DDTestSpecParsFilter(const ParameterSet& iConfig)
      : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")),
        m_attribute(iConfig.getUntrackedParameter<string>("attribute", "")),
        m_value(iConfig.getUntrackedParameter<string>("value", "")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
  const string m_attribute;
  const string m_value;
};

void DDTestSpecParsFilter::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestSpecParsFilter::analyze: " << m_tag;
  ESTransientHandle<dd4hep::SpecParRegistry> registry;
  iEventSetup.get<DDSpecParRegistryRcd>().get(m_tag, registry);

  LogVerbatim("Geometry") << "DDTestSpecParsFilter::analyze: " << m_tag << " for attribute " << m_attribute
                          << " and value " << m_value;
  LogVerbatim("Geometry") << "DD SpecPar Registry size: " << registry->specpars.size();
  std::cout << "*** Check names in a path:\n";
  std::vector<std::string_view> namesInPath = registry->names("//ME11AlumFrame");
  for (auto i : namesInPath) {
    std::cout << i << "\n";
  }
  dd4hep::SpecParRefs myReg;
  if (m_value.empty())
    registry->filter(myReg, m_attribute);
  else
    registry->filter(myReg, m_attribute, m_value);

  LogVerbatim("Geometry").log([&myReg](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << myReg.size() << "\n";
    for (const auto& t : myReg) {
      log << "\nRegExps { ";
      for (const auto& ki : t.second->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t.second->spars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
      for (const auto& kl : t.second->numpars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
    }
  });
  std::cout << "*** Check names in a path after filtering:\n";
  for (const auto& it : myReg) {
    if (it.second->hasPath("//ME11AlumFrame")) {
      std::cout << it.first << "\n";
    }
  }
}

DEFINE_FWK_MODULE(DDTestSpecParsFilter);
