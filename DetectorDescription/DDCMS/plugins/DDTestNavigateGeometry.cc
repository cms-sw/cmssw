#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepRootPersistency.h"
#include "DD4hep/DetectorTools.h"
#include "DD4hep/VolumeProcessor.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cms;
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
      cout << "Hierarchical level:" << level << "   Placement:";
      for(const auto& i : m_volumes) cout << "/" << i;
      cout << "\n\tMaterial:" << vol.material().name()
	   << "\tSolid:   " << vol.solid().name()
	   << " [" << vol.solid()->IsA()->GetName() << "]\n";
      ++m_count;
      return 1;
    }
    int count() const { return m_count; }
    
  private:
    int m_count = 0;
    vector<string> m_volumes;
  };
}

class DDTestNavigateGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DDTestNavigateGeometry(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

DDTestNavigateGeometry::DDTestNavigateGeometry(const edm::ParameterSet& iConfig)
{}

void
DDTestNavigateGeometry::analyze(const edm::Event&, const edm::EventSetup& iEventSetup)
{
  cout << "DDTestNavigateGeometry::analyze:\n";
  const DetectorDescriptionRcd& ddRecord = iEventSetup.get<DetectorDescriptionRcd>();
  edm::ESTransientHandle<DDDetector> ddd;
  ddRecord.get(ddd);
    
  dd4hep::Detector& detector = ddd->description();

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
  
  printout(ALWAYS,"VolumeScanner","+++ Visited a total of %d placed volumes.", proc.count());
}

DEFINE_FWK_MODULE(DDTestNavigateGeometry);
