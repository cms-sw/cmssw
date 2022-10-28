#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <sstream>

class SiStripCablingTrackerMap : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  SiStripCablingTrackerMap(const edm::ParameterSet& conf);
  ~SiStripCablingTrackerMap() override = default;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endJob() override;
  void endRun(const edm::Run& run, const edm::EventSetup& es) override{};
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;

  std::unique_ptr<TrackerMap> tkMap_detCab;  //0 for onTrack, 1 for offTrack, 2 for All
};

SiStripCablingTrackerMap::SiStripCablingTrackerMap(edm::ParameterSet const& conf) : detCablingToken_(esConsumes()) {}

void SiStripCablingTrackerMap::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  tkMap_detCab = std::make_unique<TrackerMap>("DetCabling");
}

//------------------------------------------------------------------------------------------
void SiStripCablingTrackerMap::endJob() {
  tkMap_detCab->save(true, 0, 0, "DetCabling.png");
  tkMap_detCab->print(true, 0, 0, "DetCabling");
}
//------------------------------------------------------------------------------------------

void SiStripCablingTrackerMap::analyze(const edm::Event& e, const edm::EventSetup& es) {
  const auto& detCabling = es.getData(detCablingToken_);
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // get list of active detectors from SiStripDetCabling
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  std::vector<uint32_t> vdetId_;
  detCabling.addActiveDetectorsRawIds(vdetId_);
  for (std::vector<uint32_t>::const_iterator detid_iter = vdetId_.begin(); detid_iter != vdetId_.end(); detid_iter++) {
    uint32_t detid = *detid_iter;
    tkMap_detCab->fill(detid, 1);
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiStripCablingTrackerMap);
