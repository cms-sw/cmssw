#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDVolumeProcessor.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepRootPersistency.h"
#include "DD4hep/DetectorTools.h"
#include "DD4hep/VolumeProcessor.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

class DDTestNavigateGeometry : public one::EDAnalyzer<> {
public:
  explicit DDTestNavigateGeometry(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
  const string m_detElementPath;
  const string m_placedVolPath;
};

DDTestNavigateGeometry::DDTestNavigateGeometry(const ParameterSet& iConfig)
    : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")),
      m_detElementPath(iConfig.getParameter<string>("detElementPath")),
      m_placedVolPath(iConfig.getParameter<string>("placedVolumePath")) {}

void DDTestNavigateGeometry::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "\nDDTestNavigateGeometry::analyze: " << m_tag;

  const DDVectorRegistryRcd& regRecord = iEventSetup.get<DDVectorRegistryRcd>();
  ESTransientHandle<DDVectorRegistry> reg;
  regRecord.get(m_tag, reg);

  LogVerbatim("Geometry").log([&reg](auto& log) {
    for (const auto& p : reg->vectors) {
      log << "\n " << p.first << " => ";
      for (const auto& i : p.second)
        log << i << ", ";
    }
  });

  const auto& ddRecord = iEventSetup.get<IdealGeometryRecord>();
  ESTransientHandle<DDDetector> det;
  ddRecord.get(m_tag, det);

  DetElement startDetEl, world = det->world();
  LogVerbatim("Geometry") << "World placement path " << world.placementPath() << ", path " << world.path();
  PlacedVolume startPVol = world.placement();
  if (!m_detElementPath.empty()) {
    LogVerbatim("Geometry") << "Det element path is " << m_detElementPath;
    startDetEl = det->findElement(m_detElementPath);
    if (startDetEl.isValid())
      LogVerbatim("Geometry") << "Found starting DetElement!\n";
  } else if (!m_placedVolPath.empty()) {
    LogVerbatim("Geometry") << "Placed volume path is " << m_placedVolPath;
    startPVol = dd4hep::detail::tools::findNode(world.placement(), m_placedVolPath);
    if (startPVol.isValid())
      LogVerbatim("Geometry") << "Found srarting PlacedVolume!\n";
  }
  if (!startPVol.isValid()) {
    if (!startDetEl.isValid()) {
      except("VolumeScanner", "Failed to find start conditions for the volume scan");
    }
    startPVol = startDetEl.placement();
  }

  DDVolumeProcessor proc;
  LogVerbatim("Geometry") << startPVol.name();
  PlacedVolumeScanner().scanPlacements(proc, startPVol, 0, true);

  LogVerbatim("Geometry") << "VolumeScanner"
                          << "+++ Visited a total of %d placed volumes." << proc.count();
}

DEFINE_FWK_MODULE(DDTestNavigateGeometry);
