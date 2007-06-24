#include <EventFilter/EcalRawToDigiDev/interface/EcalDCCHeaderRuntypeDecoder.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iostream>

EcalDCCHeaderRuntypeDecoder::EcalDCCHeaderRuntypeDecoder(){;}
EcalDCCHeaderRuntypeDecoder::~EcalDCCHeaderRuntypeDecoder(){;}

bool EcalDCCHeaderRuntypeDecoder::Decode( ulong TrigType,             // dcc header word 1, bits 56-59
					  ulong detTrigType,      // dcc header word 3, bits 0-31
					  ulong runType,             // dcc header word 3, bits 32-65
					  EcalDCCHeaderBlock* EcalDCCHeaderInfos)
{
  
  //  uint DCCNumberMask   = 63;//2^6-1

  uint WhichHalfOffSet    = 64;//2^6 
  uint TypeOffSet            = 256;//2^8
  uint SubTypeOffSet     = 2048;//2^11
  uint SettingOffSet         = 131072;//2^17;
  uint GainModeOffSet    = 16384;//2^14
  
  uint TwoBitsMask = 3;
  uint ThreeBitsMask = 7;
  uint ThirdBitMask = 4;
  
  EcalDCCHeaderInfos-> setRtHalf( int ((runType / WhichHalfOffSet) & TwoBitsMask) );
  int type           = int ((runType / TypeOffSet)      & ThreeBitsMask);
  int sequence  = int ((runType / SubTypeOffSet)   & ThreeBitsMask);
  EcalDCCHeaderInfos->setMgpaGain(int ((runType / GainModeOffSet)  & TwoBitsMask) );
  EcalDCCHeaderInfos->setMemGain( int ((runType / GainModeOffSet)  & ThirdBitMask)/ThirdBitMask );
  //  EcalDCCHeaderInfos.Setting       = int ( runType / SettingOffSet);

  
  if (type ==0 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMIC);}
  // begin: added for XDAQ 3
  else if (type ==0 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMIC);}
  else if (type ==0 && sequence == 2){
    EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::BEAMH4);}
  else if (type ==0 && sequence == 3){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::BEAMH2);}
  else if (type ==0 && sequence == 4){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::MTCC);}
  // end: added for XDAQ 3
  else if (type ==1 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_STD);}
  else if (type ==1 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_POWER_SCAN);}
  else if (type ==1 && sequence == 2){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LASER_DELAY_SCAN);}
  else if (type ==2 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM);}
  else if (type ==2 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::TESTPULSE_MGPA);}
  else if (type ==3 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_STD);}
  else if (type ==3 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN);}
  else if (type ==3 && sequence == 2){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN);}
  else if (type ==4 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::LED_STD);}

  else if (type ==5 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PHYSICS_GLOBAL);}
  else if (type ==5 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMICS_GLOBAL);}
  else if (type ==5 && sequence == 2){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::HALO_GLOBAL);}

  else if (type ==6 && sequence == 0){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::PHYSICS_LOCAL);}
  else if (type ==6 && sequence == 1){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::COSMICS_LOCAL);}
  else if (type ==6 && sequence == 2){EcalDCCHeaderInfos->setRunType(EcalDCCHeaderBlock::HALO_LOCAL);}

  else {
    edm::LogError("EcalDCCHeaderRuntypeDecoder") <<"Unrecognized runtype and sequence: "<<type<<" "<<sequence;
    EcalDCCHeaderInfos->setRunType(-1);
    WasDecodingOk_ = false;
    return WasDecodingOk_;
  }
  

  // decoding of settings depends on whether run is global or local
  if (type == 5){    DecodeSettingGlobal ( TrigType, detTrigType, EcalDCCHeaderInfos );         }
  else {                 DecodeSetting            (int ( runType / SettingOffSet),EcalDCCHeaderInfos);  }
  
  
  return WasDecodingOk_;
}





void EcalDCCHeaderRuntypeDecoder::DecodeSettingGlobal ( ulong TrigType, ulong detTrigType,  EcalDCCHeaderBlock * theHeader ){

  // if trigger is not in the gap
  if         (TrigType == 1) return;

  // if calibration trigger (gap)
  else if (TrigType == 2) {

    EcalDCCHeaderBlock::EcalDCCEventSettings theSettings;
    CleanEcalDCCSettingsInfo(&theSettings);
    
    //    int dccIdFromTCCCommand                        = (detTrigType >> H_DCCID_B)      & H_DCCID_MASK;
    int halfFromTCCCommand                            = (detTrigType >> H_HALF_B)        &  H_HALF_MASK;
    int detailedTriggerTypefromTCCCommand = (detTrigType >> H_TR_TYPE_B) &  H_TR_TYPE_MASK;
    int wavelengthFromTCCCommand               = (detTrigType >> H_WAVEL_B)    &  H_WAVEL_MASK;
    
    theHeader->                    setRtHalf( halfFromTCCCommand );
    
    if (detailedTriggerTypefromTCCCommand == 1){
      theHeader->                    setRunType(EcalDCCHeaderBlock::LASER_GAP);
      theSettings.wavelength = wavelengthFromTCCCommand;    }
    
    if (detailedTriggerTypefromTCCCommand == 2){
      theHeader->                    setRunType(EcalDCCHeaderBlock::TESTPULSE_GAP);     }
    
    if (detailedTriggerTypefromTCCCommand == 3){
      theHeader->                    setRunType(EcalDCCHeaderBlock::PEDESTAL_GAP);     }
    
    if (detailedTriggerTypefromTCCCommand == 4){
      theHeader->                    setRunType(EcalDCCHeaderBlock::LED_GAP);      }
    
    theHeader->setEventSettings(theSettings);

  }
  
  else {
    edm::LogError("EcalDCCHeaderRuntypeDecoder") <<"Unrecognized detailed trigger type";
    theHeader->setRunType(-1);
    WasDecodingOk_ = false;
  }

}



void  EcalDCCHeaderRuntypeDecoder::DecodeSetting ( int Setting,  EcalDCCHeaderBlock* theHeader )
{
  EcalDCCHeaderBlock::EcalDCCEventSettings theSettings;
  CleanEcalDCCSettingsInfo(&theSettings);

  if( theHeader->getRunType() == EcalDCCHeaderBlock::COSMIC || 
      theHeader->getRunType() == EcalDCCHeaderBlock::BEAMH2 || 
      theHeader->getRunType() == EcalDCCHeaderBlock::BEAMH4 || 
      theHeader->getRunType() == EcalDCCHeaderBlock::MTCC ||
      theHeader->getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ||
      theHeader->getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
      theHeader->getRunType() == EcalDCCHeaderBlock::HALO_LOCAL )
    {;}
  //no settings foreseen

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::LASER_STD)
    {
      theSettings.LaserPower = (Setting & 8128)/64;
      theSettings.LaserFilter = (Setting & 56)/8;
      theSettings.wavelength = Setting & 7;
    }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::LASER_POWER_SCAN){
    theSettings.LaserPower = (Setting & 8128)/64;
    theSettings.LaserFilter = (Setting & 56)/8;
    theSettings.wavelength = Setting & 7;
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::LASER_DELAY_SCAN){
    theSettings.wavelength = Setting & 7;  
    theSettings.delay = (Setting & 2040)/8;
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM){
    theSettings.MEMVinj = Setting & 511;
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA){
    theSettings.mgpa_content =  Setting & 255;
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD){;}//no settings foreseen

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN){
    theSettings.ped_offset  =  Setting;
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN ){
    theSettings.delay = (Setting & 255);  
  }

  else if(theHeader->getRunType() == EcalDCCHeaderBlock::LED_STD){
    theSettings.wavelength = Setting & 7;
  }
  else {
    edm::LogError("EcalDCCHeaderRuntypeDecoder") <<"Unrecognized run type: "<<theHeader->getRunType();
    WasDecodingOk_ = false;
  }

  theHeader->setEventSettings(theSettings);
  
}



void EcalDCCHeaderRuntypeDecoder::CleanEcalDCCSettingsInfo( EcalDCCHeaderBlock::EcalDCCEventSettings * dummySettings){
  dummySettings->LaserPower =-1;
  dummySettings->LaserFilter =-1;
  dummySettings->wavelength =-1;
  dummySettings->delay =-1;
  dummySettings->MEMVinj =-1;
  dummySettings->mgpa_content =-1;
  dummySettings->ped_offset =-1;
}
