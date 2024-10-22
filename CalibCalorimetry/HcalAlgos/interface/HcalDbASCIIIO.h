//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbASCIIIO_h
#define HcalDbASCIIIO_h

#include <iostream>
#include <memory>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationsSet.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidthsSet.h"

/**
   \class HcalDbASCIIIO
   \brief IO for ASCII instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   
Text file formats for different data types is as following:
- # in first column comments the line
- HcalPedestals, HcalGains, HcalGainWidths have identical formats:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_value(float) cap2_value(float) cap3_value(float) cap4_value(float)  HcalDetId(int,optional)
- HcalPFCuts:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) noiseThreshold(float) seedThreshold(float)
- HcalPedestalWidths:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) sigma_1_1(float) sigma_2_1 sigma_2_2 sigma_3_1 sigma_3_2 sigma_3_3 sigma_4_1 sigma_4_2 sigma_4_3 sigma_4_4
- HcalQIEShape:
  33 x floats - low edges for first 33 channels of ideal QIE
- HcalQIEData:
  eta phi depth det 4x offsets_cap1 4x offsets_cap2 4x offsets_cap3 4x offsets_cap4 4x slopes_cap1 4x slopes_cap2 4x slopes_cap3 4x slopes_cap4
- HcalChannelQuality:
  eta phi depth det status(GOOD/BAD/HOT/DEAD)
- HcalElectronicsMap:
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan subdet(HB/HE/HF/HO/HT) eta phi depth
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan "CBOX" 
                                 sector(HBM/HBP/HEM/HEP/HO0/HO1P/HO1M/HO2P/HO2M/HFP/HFM) rbx#(wage) channel
  calibration channel type association see HcalCalibDetId.h
  if electronics channel is known to be unconnected, either "subdet" or "eta" should be NA
- HcalDcsMap:
  line# Ring Slice Subchannel Type Subdetector Eta Phi Depth
- HcalFrontEndMap:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) RM# RBX#
- HcalSiPMParameters:
 eta phi depth det fcByPE darkCurrent auxi1 auxi2
- HcalSiPMCharacteristics:
 type pixels non-linearityParameters(3) auxi1 auxi2 
- HcalTPParameters
 HBHE-FGAlgorithm HF-ADCThreshold HF-TDCMask HF-SelfTriggerBits auxi1 auxi2
- HcalTPChannelParameters
 eta(int)  phi(int) depth(int) det(HB,HE,HF) Mask FGBitInfo auxi1 auxi2
- HcalCalibrationsSet (dump-only)
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_ped(float) cap2_ped(float) cap3_ped(float) cap4_ped(float) cap1_respcorrgain(float) cap2_respcorrgain(float) cap3_respcorrgain(float) cap4_respcorrgain(float) HcalDetId(int,optional)
- HcalCalibrationWidthsSet (dump-only)
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_pedw(float) cap2_pedw(float) cap3_pedw(float) cap4_pedw(float) cap1_gainw(float) cap2_gainw(float) cap3_gainw(float) cap4_gainw(float) HcalDetId(int,optional)
*/
namespace HcalDbASCIIIO {
  //alternate function for creating certain objects
  template <class T>
  std::unique_ptr<T> createObject(std::istream& fInput) {
    assert(0);  //no general case, relies on specializations defined in cc file
    return std::make_unique<T>();
  }

  bool getObject(std::istream& fInput, HcalPedestals* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalPedestals& fObject);
  bool getObject(std::istream& fInput, HcalPedestalWidths* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalPedestalWidths& fObject);
  bool getObject(std::istream& fInput, HcalGains* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalGains& fObject);
  bool getObject(std::istream& fInput, HcalGainWidths* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalGainWidths& fObject);
  bool getObject(std::istream& fInput, HcalPFCuts* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalPFCuts& fObject);
  bool getObject(std::istream& fInput, HcalQIEData* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalQIEData& fObject);
  bool getObject(std::istream& fInput, HcalCalibrationQIEData* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalCalibrationQIEData& fObject);
  bool getObject(std::istream& fInput, HcalQIETypes* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalQIETypes& fObject);
  template <>
  std::unique_ptr<HcalElectronicsMap> createObject<HcalElectronicsMap>(std::istream& fInput);
  bool dumpObject(std::ostream& fOutput, const HcalElectronicsMap& fObject);
  bool getObject(std::istream& fInput, HcalChannelQuality* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalChannelQuality& fObject);
  bool getObject(std::istream& fInput, HcalRespCorrs* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalRespCorrs& fObject);
  bool getObject(std::istream& fInput, HcalLUTCorrs* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalLUTCorrs& fObject);
  bool getObject(std::istream& fInput, HcalPFCorrs* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalPFCorrs& fObject);
  bool getObject(std::istream& fInput, HcalTimeCorrs* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalTimeCorrs& fObject);
  bool getObject(std::istream& fInput, HcalZSThresholds* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalZSThresholds& fObject);
  bool getObject(std::istream& fInput, HcalL1TriggerObjects* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalL1TriggerObjects& fObject);
  template <>
  std::unique_ptr<HcalFrontEndMap> createObject<HcalFrontEndMap>(std::istream& fInput);
  bool dumpObject(std::ostream& fOutput, const HcalFrontEndMap& fObject);

  bool getObject(std::istream& fInput, HcalValidationCorrs* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalValidationCorrs& fObject);
  bool getObject(std::istream& fInput, HcalLutMetadata* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalLutMetadata& fObject);
  bool getObject(std::istream& fInput, HcalDcsValues* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalDcsValues& fObject);
  template <>
  std::unique_ptr<HcalDcsMap> createObject<HcalDcsMap>(std::istream& fInput);
  bool dumpObject(std::ostream& fOutput, const HcalDcsMap& fObject);

  bool getObject(std::istream& fInput, HcalRecoParams* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalRecoParams& fObject);
  bool getObject(std::istream& fInput, HcalLongRecoParams* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalLongRecoParams& fObject);

  bool getObject(std::istream& fInput, HcalZDCLowGainFractions* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalZDCLowGainFractions& fObject);

  bool getObject(std::istream& fInput, HcalTimingParams* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalTimingParams& fObject);

  bool getObject(std::istream& fInput, HcalMCParams* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalMCParams& fObject);

  // Getting/Dumping Hcal Flag information
  bool getObject(std::istream& fInput, HcalFlagHFDigiTimeParams* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalFlagHFDigiTimeParams& fObject);

  bool getObject(std::istream& fInput, HcalSiPMParameters* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalSiPMParameters& fObject);
  template <>
  std::unique_ptr<HcalSiPMCharacteristics> createObject<HcalSiPMCharacteristics>(std::istream& fInput);
  bool dumpObject(std::ostream& fOutput, const HcalSiPMCharacteristics& fObject);

  bool getObject(std::istream& fInput, HcalTPParameters* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalTPParameters& fObject);
  bool getObject(std::istream& fInput, HcalTPChannelParameters* fObject);
  bool dumpObject(std::ostream& fOutput, const HcalTPChannelParameters& fObject);

  bool dumpObject(std::ostream& fOutput, const HcalCalibrationsSet& fObject);
  bool dumpObject(std::ostream& fOutput, const HcalCalibrationWidthsSet& fObject);

  DetId getId(const std::vector<std::string>& items);
  void dumpId(std::ostream& fOutput, DetId id);
  void dumpIdShort(std::ostream& fOutput, DetId id);
}  // namespace HcalDbASCIIIO
#endif
