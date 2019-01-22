#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
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

namespace {
  
  class VolumeProcessor : public dd4hep::PlacedVolumeProcessor {
  public:
    VolumeProcessor() = default;
    ~VolumeProcessor() override = default;
    
    /// Callback to retrieve PlacedVolume information of an entire Placement
    int process(dd4hep::PlacedVolume pv, int level, bool recursive) override {
      m_volumes.emplace_back(pv.name());
      int ret = dd4hep::PlacedVolumeProcessor::process(pv, level, recursive);
      m_volumes.pop_back();
      return ret;
    }
    
    /// Volume callback
    int operator()(dd4hep::PlacedVolume pv, int level) override {
      dd4hep::Volume vol = pv.volume();
      LogVerbatim("Geometry").log([&level, &vol, this](auto& log) {
	  log << "\nHierarchical level:" << level << "   Placement:";
	  for(const auto& i : m_volumes)
	    log << "/" << i;
	  log << "\n\tMaterial:" << vol.material().name()
	      << "\tSolid:   " << vol.solid().name()
	      << " [" << vol.solid()->IsA()->GetName() << "]\n";
	});
      ++m_count;
      return 1;
    }
    int count() const { return m_count; }
    
  private:
    int m_count = 0;
    vector<string> m_volumes;
  };
}

class DDTestNavigateGeometry : public one::EDAnalyzer<> {
public:
  explicit DDTestNavigateGeometry(const ParameterSet&);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:  
  const ESInputTag m_tag;
};

DDTestNavigateGeometry::DDTestNavigateGeometry(const ParameterSet& iConfig)
  : m_tag(iConfig.getParameter<ESInputTag>("DDDetector"))
{}

void
DDTestNavigateGeometry::analyze(const Event&, const EventSetup& iEventSetup)
{
  LogVerbatim("Geometry") << "\nDDTestNavigateGeometry::analyze: " << m_tag;

  const DDVectorRegistryRcd& regRecord = iEventSetup.get<DDVectorRegistryRcd>();
  ESTransientHandle<DDVectorRegistry> reg;
  regRecord.get(m_tag.module(), reg);

  LogVerbatim("Geometry").log([&reg](auto& log) {
      for(const auto& p: reg->vectors) {
	log << "\n " << p.first << " => ";
	for(const auto& i : p.second)
	  log << i << ", ";
      }
    });
  
  const DetectorDescriptionRcd& ddRecord = iEventSetup.get<DetectorDescriptionRcd>();
  ESTransientHandle<DDDetector> ddd;
  ddRecord.get(m_tag.module(), ddd);
    
  const dd4hep::Detector& detector = *ddd->description();

  string detElementPath;
  string placedVolPath;
 
  DetElement startDetEl, world = detector.world();
  PlacedVolume startPVol = world.placement();
  if( !detElementPath.empty())
    startDetEl = dd4hep::detail::tools::findElement(detector, detElementPath);
  else if( !placedVolPath.empty())
    startPVol = dd4hep::detail::tools::findNode(world.placement(), placedVolPath);
  if( !startPVol.isValid()) {
    if( !startDetEl.isValid()) {      
      except("VolumeScanner", "Failed to find start conditions for the volume scan");
    }
    startPVol = startDetEl.placement();
  }

  VolumeProcessor proc;
  PlacedVolumeScanner().scanPlacements(proc, startPVol, 0, true);
  
  LogVerbatim("Geometry") << "VolumeScanner" << "+++ Visited a total of %d placed volumes." << proc.count();
}

DEFINE_FWK_MODULE(DDTestNavigateGeometry);
