#include "FWCore/Framework/interface/global/EDAnalyzer.h"
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

class DDTestVectors : public global::EDAnalyzer<> {
public:
  explicit DDTestVectors(const ParameterSet& iConfig)
      : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")), m_token(esConsumes(m_tag)) {}

  void beginJob() override {}
  void analyze(StreamID, Event const& iEvent, EventSetup const&) const override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
  const ESGetToken<DDVectorRegistry, DDVectorRegistryRcd> m_token;
};

void DDTestVectors::analyze(StreamID, const Event&, const EventSetup& iEventSetup) const {
  ESTransientHandle<DDVectorRegistry> registry = iEventSetup.getTransientHandle(m_token);

  LogVerbatim("Geometry").log([&registry, this](auto& log) {
    log << "DDTestVectors::analyze: " << m_tag;
    log << "DD Vector Registry size: " << registry->vectors.size() << "\n";
    for (const auto& p : registry->vectors) {
      log << " " << p.first << " => ";
      for (const auto& i : p.second)
        log << i << ", ";
      log << "\n";
    }
  });
}

DEFINE_FWK_MODULE(DDTestVectors);
