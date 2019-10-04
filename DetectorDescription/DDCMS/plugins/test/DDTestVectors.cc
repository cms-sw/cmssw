#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestVectors : public one::EDAnalyzer<> {
public:
  explicit DDTestVectors(const ParameterSet& iConfig) : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
};

void DDTestVectors::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestVectors::analyze: " << m_tag;
  ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(m_tag, registry);

  LogVerbatim("Geometry").log([&registry](auto& log) {
    log << "DD Vector Registry size: " << registry->vectors.size();
    for (const auto& p : registry->vectors) {
      log << " " << p.first << " => ";
      for (const auto& i : p.second)
        log << i << ", ";
    }
  });
}

DEFINE_FWK_MODULE(DDTestVectors);
