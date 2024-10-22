#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCHeaderRuntypeDecoder.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iostream>

EcalDCCTBHeaderRuntypeDecoder::EcalDCCTBHeaderRuntypeDecoder() { ; }
EcalDCCTBHeaderRuntypeDecoder::~EcalDCCTBHeaderRuntypeDecoder() { ; }

bool EcalDCCTBHeaderRuntypeDecoder::Decode(unsigned long headerWord, EcalDCCHeaderBlock* EcalDCCHeaderInfos) {
  //  unsigned long DCCNumberMask      = 63;//2^6-1

  unsigned long WhichHalfOffSet = 64;    //2^6
  unsigned long TypeOffSet = 256;        //2^8
  unsigned long SubTypeOffSet = 2048;    //2^11
  unsigned long SettingOffSet = 131072;  //2^17;
  unsigned long GainModeOffSet = 16384;  //2^14

  unsigned long TwoBitsMask = 3;
  unsigned long ThreeBitsMask = 7;
  unsigned long ThirdBitMask = 4;

  //  EcalDCCTBHeaderInfos->setId( int ( headerWord & DCCNumberMask) );
  EcalDCCHeaderInfos->setRtHalf(int((headerWord / WhichHalfOffSet) & TwoBitsMask));
  int type = int((headerWord / TypeOffSet) & ThreeBitsMask);
  int sequence = int((headerWord / SubTypeOffSet) & ThreeBitsMask);
  EcalDCCHeaderInfos->setMgpaGain(int((headerWord / GainModeOffSet) & TwoBitsMask));
  EcalDCCHeaderInfos->setMemGain(int((headerWord / GainModeOffSet) & ThirdBitMask) / ThirdBitMask);
  //  EcalDCCHeaderInfos.Setting       = int ( headerWord / SettingOffSet);

  if (type == 0 && sequence == 0) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMIC);
  }
  // begin: added for XDAQ 3
  else if (type == 0 && sequence == 1) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMIC);
  } else if (type == 0 && sequence == 2) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::BEAMH4);
  } else if (type == 0 && sequence == 3) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::BEAMH2);
  } else if (type == 0 && sequence == 4) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::MTCC);
  }
  // end: added for XDAQ 3
  else if (type == 1 && sequence == 0) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_STD);
  } else if (type == 1 && sequence == 1) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_POWER_SCAN);
  } else if (type == 1 && sequence == 2) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_DELAY_SCAN);
  } else if (type == 2 && sequence == 0) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM);
  } else if (type == 2 && sequence == 1) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::TESTPULSE_MGPA);
  } else if (type == 3 && sequence == 0) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_STD);
  } else if (type == 3 && sequence == 1) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN);
  } else if (type == 3 && sequence == 2) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN);
  } else if (type == 4 && sequence == 0) {
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LED_STD);
  } else {
    edm::LogWarning("EcalTBRawToDigi") << "@SUB=EcalDCCHeaderRuntypeDecoder::Decode unrecognized runtype and sequence: "
                                       << type << " " << sequence;
    EcalDCCHeaderInfos->setRunType(-1);
    WasDecodingOk_ = false;
  }

  DecodeSetting(int(headerWord / SettingOffSet), EcalDCCHeaderInfos);

  return WasDecodingOk_;
}

void EcalDCCTBHeaderRuntypeDecoder::DecodeSetting(int Setting, EcalDCCHeaderBlock* theHeader) {
  EcalDCCHeaderBlock::EcalDCCEventSettings theSettings;  // = new EcalDCCEventSettings;
  CleanEcalDCCSettingsInfo(&theSettings);

  if (theHeader->getRunType() == EcalDCCHeaderBlock::COSMIC || theHeader->getRunType() == EcalDCCHeaderBlock::BEAMH2 ||
      theHeader->getRunType() == EcalDCCHeaderBlock::BEAMH4 || theHeader->getRunType() == EcalDCCHeaderBlock::MTCC) {
    ;
  }  //no settings foreseen
  else if (theHeader->getRunType() == EcalDCCHeaderBlock::LASER_STD) {
    theSettings.LaserPower = (Setting & 8128) / 64;
    theSettings.LaserFilter = (Setting & 56) / 8;
    theSettings.wavelength = Setting & 7;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::LASER_POWER_SCAN) {
    theSettings.LaserPower = (Setting & 8128) / 64;
    theSettings.LaserFilter = (Setting & 56) / 8;
    theSettings.wavelength = Setting & 7;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::LASER_DELAY_SCAN) {
    theSettings.wavelength = Setting & 7;
    theSettings.delay = (Setting & 2040) / 8;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM) {
    theSettings.MEMVinj = Setting & 511;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA) {
    theSettings.mgpa_content = Setting & 255;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD) {
    ;
  }  //no settings foreseen
  else if (theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN) {
    theSettings.ped_offset = Setting;
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN) {
    theSettings.delay = (Setting & 255);
  } else if (theHeader->getRunType() == EcalDCCHeaderBlock::LED_STD) {
    theSettings.wavelength = Setting & 7;
  } else {
    edm::LogWarning("EcalTBRawToDigi") << "@SUB=EcalDCCTBHeaderRuntypeDecoder::DecodeSettings unrecognized run type: "
                                       << theHeader->getRunType();
    WasDecodingOk_ = false;
  }
  theHeader->setEventSettings(theSettings);
}

void EcalDCCTBHeaderRuntypeDecoder::CleanEcalDCCSettingsInfo(EcalDCCHeaderBlock::EcalDCCEventSettings* dummySettings) {
  dummySettings->LaserPower = -1;
  dummySettings->LaserFilter = -1;
  dummySettings->wavelength = -1;
  dummySettings->delay = -1;
  dummySettings->MEMVinj = -1;
  dummySettings->mgpa_content = -1;
  dummySettings->ped_offset = -1;
}
