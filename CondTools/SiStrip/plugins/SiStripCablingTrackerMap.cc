#include "CondTools/SiStrip/plugins/SiStripCablingTrackerMap.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include <sstream>

SiStripCablingTrackerMap::SiStripCablingTrackerMap(edm::ParameterSet const& conf) : conf_(conf) {}

SiStripCablingTrackerMap::~SiStripCablingTrackerMap() {}

void SiStripCablingTrackerMap::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
  tkMap_detCab = new TrackerMap("DetCabling");
}

//------------------------------------------------------------------------------------------

void SiStripCablingTrackerMap::endJob() {
  tkMap_detCab->save(true, 0, 0, "DetCabling.png");
  tkMap_detCab->print(true, 0, 0, "DetCabling");
}
//------------------------------------------------------------------------------------------

void SiStripCablingTrackerMap::analyze(const edm::Event& e, const edm::EventSetup& es) {
  es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // get list of active detectors from SiStripDetCabling
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  std::vector<uint32_t> vdetId_;
  SiStripDetCabling_->addActiveDetectorsRawIds(vdetId_);
  for (unsigned int detid : vdetId_) {
    tkMap_detCab->fill(detid, 1);
  }
}
