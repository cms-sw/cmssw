#ifndef CastorDbHardcodeIn_h
#define CastorDbHardcodeIn_h

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidth.h"
#include "CondFormats/CastorObjects/interface/CastorGain.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidth.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorCalibrationQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParam.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorr.h"

namespace CastorDbHardcode {
  CastorPedestal makePedestal (HcalGenericDetId fId, bool fSmear = false);
  CastorPedestalWidth makePedestalWidth (HcalGenericDetId fId);
  CastorGain makeGain (HcalGenericDetId fId, bool fSmear = false);
  CastorGainWidth makeGainWidth (HcalGenericDetId fId);
  CastorQIECoder makeQIECoder (HcalGenericDetId fId);
  CastorCalibrationQIECoder makeCalibrationQIECoder (HcalGenericDetId fId);
  CastorQIEShape makeQIEShape ();
  CastorRecoParam makeRecoParam (HcalGenericDetId fId);
  CastorSaturationCorr makeSaturationCorr (HcalGenericDetId fId);
  void makeHardcodeMap(CastorElectronicsMap& emap);
}
#endif
