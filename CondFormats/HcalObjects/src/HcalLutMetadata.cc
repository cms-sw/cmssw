/**
\class HcalLutMetadata
\author Gena Kukartsev
POOL object to store conditions associated with HCAL trigger primitive LUT 
$Author: kukartse
$Date: 2009/09/18 14:24:16 $
$Revision: 1.1 $
*/

#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"

bool HcalLutMetadata::setRctLsb(float rctlsb){
  bool result=false;
  if (rctlsb==0.25 || rctlsb==0.5){
    mNonChannelData.mRctLsb=rctlsb;
    result=true;
  }
  return result;
}


bool HcalLutMetadata::setNominalGain(float gain){
  bool result = false;
  mNonChannelData.mNominalGain=gain;
  result=true;
  return result;
}
