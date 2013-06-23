//
// F.Ratnikov (UMd), Dec. 14, 2005
//
#ifndef HcalDbHardcodeIn_h
#define HcalDbHardcodeIn_h

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "CondFormats/HcalObjects/interface/HcalGain.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidth.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "CondFormats/HcalObjects/interface/HcalDcsMap.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalTimingParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"

/**

   \class HcalDbHardcode
   \brief Hardcode implementation of some conditions data
   \author Fedor Ratnikov
   
*/
namespace HcalDbHardcode {
  HcalPedestal makePedestal (HcalGenericDetId fId, bool fSmear = false);
  HcalPedestal makePedestal (HcalGenericDetId fId, bool fSmear = false, double lumi = 0.0);
  HcalPedestalWidth makePedestalWidth (HcalGenericDetId fId);
  HcalPedestalWidth makePedestalWidth (HcalGenericDetId fId, double lumi);
  HcalGain makeGain (HcalGenericDetId fId, bool fSmear = false);
  HcalGainWidth makeGainWidth (HcalGenericDetId fId);
  HcalQIECoder makeQIECoder (HcalGenericDetId fId);
  HcalCalibrationQIECoder makeCalibrationQIECoder (HcalGenericDetId fId);
  HcalQIEShape makeQIEShape ();
  HcalRecoParam makeRecoParam (HcalGenericDetId fId);
  HcalMCParam makeMCParam (HcalGenericDetId fId);
  HcalTimingParam makeTimingParam (HcalGenericDetId fId);
  void makeHardcodeMap(HcalElectronicsMap& emap);
  void makeHardcodeDcsMap(HcalDcsMap& dcs_map);
}
#endif
