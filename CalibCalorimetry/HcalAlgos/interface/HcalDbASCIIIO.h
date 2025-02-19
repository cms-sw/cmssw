//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbASCIIIO_h
#define HcalDbASCIIIO_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

/**
   \class HcalDbASCIIIO
   \brief IO for ASCII instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbASCIIIO.h,v 1.21 2011/11/23 13:48:27 abdullin Exp $
   
Text file formats for different data types is as following:
- # in first column comments the line
- HcalPedestals, HcalGains, HcalGainWidths have identical formats:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_value(float) cap2_value(float) cap3_value(float) cap4_value(float)  HcalDetId(int,optional)
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
*/
namespace HcalDbASCIIIO {
  bool getObject (std::istream& fInput, HcalPedestals* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestals& fObject);
  bool getObject (std::istream& fInput, HcalPedestalWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestalWidths& fObject);
  bool getObject (std::istream& fInput, HcalGains* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGains& fObject);
  bool getObject (std::istream& fInput, HcalGainWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGainWidths& fObject);
  bool getObject (std::istream& fInput, HcalQIEData* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalQIEData& fObject);
  bool getObject (std::istream& fInput, HcalCalibrationQIEData* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalCalibrationQIEData& fObject);
  bool getObject (std::istream& fInput, HcalElectronicsMap* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalElectronicsMap& fObject);
  bool getObject (std::istream& fInput, HcalChannelQuality* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalChannelQuality& fObject);
  bool getObject (std::istream& fInput, HcalRespCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalRespCorrs& fObject);
  bool getObject (std::istream& fInput, HcalLUTCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalLUTCorrs& fObject);
  bool getObject (std::istream& fInput, HcalPFCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPFCorrs& fObject);
  bool getObject (std::istream& fInput, HcalTimeCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalTimeCorrs& fObject);
  bool getObject (std::istream& fInput, HcalZSThresholds* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalZSThresholds& fObject);
  bool getObject (std::istream& fInput, HcalL1TriggerObjects* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalL1TriggerObjects& fObject);

  bool getObject (std::istream& fInput, HcalValidationCorrs* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalValidationCorrs& fObject);
  bool getObject (std::istream& fInput, HcalLutMetadata* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalLutMetadata& fObject);
  bool getObject (std::istream& fInput, HcalDcsValues* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalDcsValues& fObject);
  bool getObject (std::istream& fInput, HcalDcsMap* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalDcsMap& fObject);

  bool getObject (std::istream& fInput, HcalRecoParams* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalRecoParams& fObject);
  bool getObject (std::istream& fInput, HcalLongRecoParams* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalLongRecoParams& fObject);

  bool getObject (std::istream& fInput, HcalTimingParams* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalTimingParams& fObject);

  bool getObject (std::istream& fInput, HcalMCParams* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalMCParams& fObject);

  bool getObject (std::istream& fInput, HcalCholeskyMatrices* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalCholeskyMatrices& fObject);
  bool getObject (std::istream& fInput, HcalCovarianceMatrices* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalCovarianceMatrices& fObject);
  // Getting/Dumping Hcal Flag information
  bool getObject (std::istream& fInput, HcalFlagHFDigiTimeParams* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalFlagHFDigiTimeParams& fObject);

  DetId getId (const std::vector <std::string> & items);
  void dumpId (std::ostream& fOutput, DetId id);
  void dumpIdShort (std::ostream& fOutput, DetId id);
} 
#endif
