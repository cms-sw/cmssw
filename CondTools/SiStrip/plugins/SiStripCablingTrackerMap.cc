#include "CondTools/SiStrip/plugins/SiStripCablingTrackerMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <sstream>

SiStripCablingTrackerMap::SiStripCablingTrackerMap(edm::ParameterSet const& conf)
    : conf_(conf), detCablingToken_(esConsumes()) {}

SiStripCablingTrackerMap::~SiStripCablingTrackerMap() {}

void SiStripCablingTrackerMap::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  tkMap_detCab = new TrackerMap("DetCabling");
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
