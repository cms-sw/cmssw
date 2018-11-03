#ifndef CastorDbASCIIIO_h
#define CastorDbASCIIIO_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

/**
   \class CastorDbASCIIIO
   \brief IO for ASCII instances of Castor/HCAL Calibrations
   
Text file formats for different data types is as following:
- # in first column comments the line
- CastorPedestals, CastorGains, CastorGainWidths have identical formats:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) cap1_value(float) cap2_value(float) cap3_value(float) cap4_value(float)  HcalDetId(int,optional)
- CastorPedestalWidths:
  eta(int)  phi(int) depth(int) det(HB,HE,HF) sigma_1_1(float) sigma_2_1 sigma_2_2 sigma_3_1 sigma_3_2 sigma_3_3 sigma_4_1 sigma_4_2 sigma_4_3 sigma_4_4
- CastorQIEShape:
  33 x floats - low edges for first 33 channels of ideal QIE
- CastorQIEData:
  eta phi depth det 4x offsets_cap1 4x offsets_cap2 4x offsets_cap3 4x offsets_cap4 4x slopes_cap1 4x slopes_cap2 4x slopes_cap3 4x slopes_cap4
- CastorChannelQuality:
  eta phi depth det status(GOOD/BAD/HOT/DEAD)
- CastorElectronicsMap:
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan subdet(HB/HE/HF/HO/HT) eta phi depth
  line#  crate HTR_slot top_bottom(t/b) dcc# dcc_spigot fiber fiberchan "CBOX" 
                                 sector(HBM/HBP/HEM/HEP/HO0/HO1P/HO1M/HO2P/HO2M/HFP/HFM) rbx#(wage) channel
  calibration channel type association see HcalCalibDetId.h
  if electronics channel is known to be unconnected, either "subdet" or "eta" should be NA
- CastorRecoParams
  eta(int)  phi(int) depth(int) det(HB,HE,HF) firstSample(unsigned int) samplesToAdd(unsigned int)  HcalDetId(int,optional)
- CastorSaturationCorrs
  eta(int)  phi(int) depth(int) det(HB,HE,HF) SatCorr(float) HcalDetId(int,optional)
  
*/
namespace CastorDbASCIIIO {
  bool getObject (std::istream& fInput, CastorPedestals& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorPedestals& fObject);
  bool getObject (std::istream& fInput, CastorPedestalWidths& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorPedestalWidths& fObject);
  bool getObject (std::istream& fInput, CastorGains& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorGains& fObject);
  bool getObject (std::istream& fInput, CastorGainWidths& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorGainWidths& fObject);
  bool getObject (std::istream& fInput, CastorQIEData& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorQIEData& fObject);
  bool getObject (std::istream& fInput, CastorCalibrationQIEData& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorCalibrationQIEData& fObject);
  bool getObject (std::istream& fInput, CastorElectronicsMap& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorElectronicsMap& fObject);
  bool getObject (std::istream& fInput, CastorChannelQuality& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorChannelQuality& fObject);
  bool getObject (std::istream& fInput, CastorRecoParams& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorRecoParams& fObject);
  bool getObject (std::istream& fInput, CastorSaturationCorrs& fObject);
  bool dumpObject (std::ostream& fOutput, const CastorSaturationCorrs& fObject);
  DetId getId (const std::vector <std::string> & items);
  void dumpId (std::ostream& fOutput, DetId id);
} 
#endif
