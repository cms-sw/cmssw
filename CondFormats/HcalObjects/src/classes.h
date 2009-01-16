#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
namespace {
  HcalPedestals mypeds(false);
  std::vector<HcalPedestal> mypedsVec;
}

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
namespace {
  HcalPedestalWidths mywidths(false);
  std::vector<HcalPedestalWidth> mywidthsVec;
}

#include "CondFormats/HcalObjects/interface/HcalGains.h"
namespace {
  HcalGains mygains;
  std::vector<HcalGain> mygainsVec;
}

#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
namespace {
  HcalGainWidths mygwidths;
  std::vector<HcalGainWidth> mygwidthsVec;
}

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
namespace {
  HcalQIEData myqie;
  std::vector<HcalQIECoder> myqievec;
}

#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"
namespace {
  HcalCalibrationQIEData mycalqie;
  std::vector<HcalCalibrationQIECoder> mycalqieVec;
}

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
namespace {
  HcalElectronicsMap mymap;
  std::vector<HcalElectronicsMap::PrecisionItem> mymap2;
  std::vector<HcalElectronicsMap::TriggerItem> mymap3;
}

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
namespace {
  HcalChannelQuality myquality;
  std::vector<HcalChannelStatus> myqualityVec;
}

#include "CondFormats/HcalObjects/interface/HcalZSThresholds.h"
namespace {
  HcalZSThresholds myth;
  std::vector<HcalZSThreshold> mythvec;
}

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
namespace {
  HcalRespCorrs mycorrs;
  std::vector<HcalRespCorr> mycorrsVec;
}

#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"
namespace {
  HcalL1TriggerObjects myL1trigs;
  std::vector<HcalL1TriggerObject> myL1trigsVec;
}

