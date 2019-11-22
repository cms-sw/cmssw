#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

class DDCMSDetector : public one::EDAnalyzer<> {
public:
  explicit DDCMSDetector(const ParameterSet& p);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override;

private:
  const ESInputTag m_tag;
};

DDCMSDetector::DDCMSDetector(const ParameterSet& iConfig) : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

void DDCMSDetector::analyze(const Event&, const EventSetup& iEventSetup) {
  ESTransientHandle<DDDetector> det;
  iEventSetup.get<IdealGeometryRecord>().get(m_tag, det);

  LogVerbatim("Geometry") << "Iterate over the detectors:\n";
  LogVerbatim("Geometry").log([&](auto& log) {
    for (auto const& it : det->detectors()) {
      dd4hep::DetElement det(it.second);
      log << it.first << ": " << det.path();
    }
  });
  LogVerbatim("Geometry") << "..done!";

  ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(m_tag, registry);

  LogVerbatim("Geometry") << "DD Vector Registry size: " << registry->vectors.size();
  LogVerbatim("Geometry").log([&](auto& log) {
    for (const auto& p : registry->vectors) {
      log << " " << p.first << " => ";
      for (const auto& i : p.second)
        log << i << ", ";
      log << '\n';
    }
  });
}

void DDCMSDetector::endJob() {}

DEFINE_FWK_MODULE(DDCMSDetector);
