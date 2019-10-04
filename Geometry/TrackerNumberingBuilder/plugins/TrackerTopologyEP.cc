#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}

class TrackerTopologyEP : public edm::ESProducer {
public:
  TrackerTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<TrackerTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const TrackerTopologyRcd&);

private:
  void fillParameters(const PTrackerParameters&,
                      TrackerTopology::PixelBarrelValues& pxbVals,
                      TrackerTopology::PixelEndcapValues& pxfVals,
                      TrackerTopology::TECValues& tecVals,
                      TrackerTopology::TIBValues& tibVals,
                      TrackerTopology::TIDValues& tidVals,
                      TrackerTopology::TOBValues& tobVals);

  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> token_;
};

TrackerTopologyEP::TrackerTopologyEP(const edm::ParameterSet& conf)
    : token_(setWhatProduced(this).consumesFrom<PTrackerParameters, PTrackerParametersRcd>(edm::ESInputTag())) {
  edm::LogInfo("TRACKER") << "TrackerTopologyEP::TrackerTopologyEP";
}

void TrackerTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription ttc;
  descriptions.add("trackerTopology", ttc);
}

TrackerTopologyEP::ReturnType TrackerTopologyEP::produce(const TrackerTopologyRcd& iRecord) {
  edm::LogInfo("TrackerTopologyEP") << "TrackerTopologyEP::produce(const TrackerTopologyRcd& iRecord)";
  auto ptp = iRecord.getRecord<PTrackerParametersRcd>().getTransientHandle(token_);

  TrackerTopology::PixelBarrelValues pxbVals;
  TrackerTopology::PixelEndcapValues pxfVals;
  TrackerTopology::TECValues tecVals;
  TrackerTopology::TIBValues tibVals;
  TrackerTopology::TIDValues tidVals;
  TrackerTopology::TOBValues tobVals;

  fillParameters(*ptp, pxbVals, pxfVals, tecVals, tibVals, tidVals, tobVals);

  return std::make_unique<TrackerTopology>(pxbVals, pxfVals, tecVals, tibVals, tidVals, tobVals);
}

void TrackerTopologyEP::fillParameters(const PTrackerParameters& ptp,
                                       TrackerTopology::PixelBarrelValues& pxbVals,
                                       TrackerTopology::PixelEndcapValues& pxfVals,
                                       TrackerTopology::TECValues& tecVals,
                                       TrackerTopology::TIBValues& tibVals,
                                       TrackerTopology::TIDValues& tidVals,
                                       TrackerTopology::TOBValues& tobVals) {
  pxbVals.layerStartBit_ = ptp.vitems[0].vpars[0];   // 16
  pxbVals.ladderStartBit_ = ptp.vitems[0].vpars[1];  // 8
  pxbVals.moduleStartBit_ = ptp.vitems[0].vpars[2];  // 2
  pxbVals.layerMask_ = ptp.vitems[0].vpars[3];       // 0xF
  pxbVals.ladderMask_ = ptp.vitems[0].vpars[4];      // 0xFF
  pxbVals.moduleMask_ = ptp.vitems[0].vpars[5];      // 0x3F

  pxfVals.sideStartBit_ = ptp.vitems[1].vpars[0];
  pxfVals.diskStartBit_ = ptp.vitems[1].vpars[1];
  pxfVals.bladeStartBit_ = ptp.vitems[1].vpars[2];
  pxfVals.panelStartBit_ = ptp.vitems[1].vpars[3];
  pxfVals.moduleStartBit_ = ptp.vitems[1].vpars[4];
  pxfVals.sideMask_ = ptp.vitems[1].vpars[5];
  pxfVals.diskMask_ = ptp.vitems[1].vpars[6];
  pxfVals.bladeMask_ = ptp.vitems[1].vpars[7];
  pxfVals.panelMask_ = ptp.vitems[1].vpars[8];
  pxfVals.moduleMask_ = ptp.vitems[1].vpars[9];

  // TEC: 6
  tecVals.sideStartBit_ = ptp.vitems[5].vpars[0];
  tecVals.wheelStartBit_ = ptp.vitems[5].vpars[1];
  tecVals.petal_fw_bwStartBit_ = ptp.vitems[5].vpars[2];
  tecVals.petalStartBit_ = ptp.vitems[5].vpars[3];
  tecVals.ringStartBit_ = ptp.vitems[5].vpars[4];
  tecVals.moduleStartBit_ = ptp.vitems[5].vpars[5];
  tecVals.sterStartBit_ = ptp.vitems[5].vpars[6];
  tecVals.sideMask_ = ptp.vitems[5].vpars[7];
  tecVals.wheelMask_ = ptp.vitems[5].vpars[8];
  tecVals.petal_fw_bwMask_ = ptp.vitems[5].vpars[9];
  tecVals.petalMask_ = ptp.vitems[5].vpars[10];
  tecVals.ringMask_ = ptp.vitems[5].vpars[11];
  tecVals.moduleMask_ = ptp.vitems[5].vpars[12];
  tecVals.sterMask_ = ptp.vitems[5].vpars[13];

  // TIB: 3
  tibVals.layerStartBit_ = ptp.vitems[2].vpars[0];
  tibVals.str_fw_bwStartBit_ = ptp.vitems[2].vpars[1];
  tibVals.str_int_extStartBit_ = ptp.vitems[2].vpars[2];
  tibVals.strStartBit_ = ptp.vitems[2].vpars[3];
  tibVals.moduleStartBit_ = ptp.vitems[2].vpars[4];
  tibVals.sterStartBit_ = ptp.vitems[2].vpars[5];
  tibVals.layerMask_ = ptp.vitems[2].vpars[6];
  tibVals.str_fw_bwMask_ = ptp.vitems[2].vpars[7];
  tibVals.str_int_extMask_ = ptp.vitems[2].vpars[8];
  tibVals.strMask_ = ptp.vitems[2].vpars[9];
  tibVals.moduleMask_ = ptp.vitems[2].vpars[10];
  tibVals.sterMask_ = ptp.vitems[2].vpars[11];

  // TID: 4
  tidVals.sideStartBit_ = ptp.vitems[3].vpars[0];
  tidVals.wheelStartBit_ = ptp.vitems[3].vpars[1];
  tidVals.ringStartBit_ = ptp.vitems[3].vpars[2];
  tidVals.module_fw_bwStartBit_ = ptp.vitems[3].vpars[3];
  tidVals.moduleStartBit_ = ptp.vitems[3].vpars[4];
  tidVals.sterStartBit_ = ptp.vitems[3].vpars[5];
  tidVals.sideMask_ = ptp.vitems[3].vpars[6];
  tidVals.wheelMask_ = ptp.vitems[3].vpars[7];
  tidVals.ringMask_ = ptp.vitems[3].vpars[8];
  tidVals.module_fw_bwMask_ = ptp.vitems[3].vpars[9];
  tidVals.moduleMask_ = ptp.vitems[3].vpars[10];
  tidVals.sterMask_ = ptp.vitems[3].vpars[11];

  // TOB: 5
  tobVals.layerStartBit_ = ptp.vitems[4].vpars[0];
  tobVals.rod_fw_bwStartBit_ = ptp.vitems[4].vpars[1];
  tobVals.rodStartBit_ = ptp.vitems[4].vpars[2];
  tobVals.moduleStartBit_ = ptp.vitems[4].vpars[3];
  tobVals.sterStartBit_ = ptp.vitems[4].vpars[4];
  tobVals.layerMask_ = ptp.vitems[4].vpars[5];
  tobVals.rod_fw_bwMask_ = ptp.vitems[4].vpars[6];
  tobVals.rodMask_ = ptp.vitems[4].vpars[7];
  tobVals.moduleMask_ = ptp.vitems[4].vpars[8];
  tobVals.sterMask_ = ptp.vitems[4].vpars[9];
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerTopologyEP);
