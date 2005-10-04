
//
// F.Ratnikov (UMd), Sep. 30, 2005
//
#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServicePool.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"


namespace {

  unsigned int poolDetId (const cms::HcalDetId& fCell) {
    return fCell.rawId();
  }
};

HcalDbServicePool::HcalDbServicePool () {
  mDefault = new HcalDbServiceHardcode ();
}
HcalDbServicePool::~HcalDbServicePool () {
  delete mDefault;
}


const char* HcalDbServicePool::name () const {return "HcalDbServicePool";}

// basic conversion function for single range (0<=count<32)
double HcalDbServicePool::adcShape (int fCount) const {
  std::cerr << "HcalDbServicePool::adcShape is not implemented. Forward to HcalDbServiceHardcode" << std::endl;
  return mDefault->adcShape (fCount);
}
// bin size for the QIE conversion
double HcalDbServicePool::adcShapeBin (int fCount) const {
  std::cerr << "HcalDbServicePool::adcShapeBin is not implemented. Forward to HcalDbServiceHardcode" << std::endl;
  return mDefault->adcShapeBin (fCount);
}
  // pedestal  
const float* HcalDbServicePool::pedestals (const cms::HcalDetId& fCell) const {
  if (!mPedestals) {
    std::cerr << "HcalDbServicePool::pedestals-> Pedestals are not defined. Use defaults" << std::endl;
    return mDefault->pedestals (fCell);
  }
  return mPedestals->getValues (poolDetId (fCell));
}
  // pedestal width
const float* HcalDbServicePool::pedestalErrors (const cms::HcalDetId& fCell) const {
  if (!mPedestalWidths) {
    std::cerr << "HcalDbServicePool::pedestalErrors-> Pedestals widths are not defined. Use defaults" << std::endl;
    return mDefault->pedestalErrors (fCell); 
  }
  return mPedestalWidths->getValues (poolDetId (fCell));
}
  // gain
const float* HcalDbServicePool::gains (const cms::HcalDetId& fCell) const {
  if (!mGains) {
    std::cerr << "HcalDbServicePool::gains-> Gains are not defined. Use defaults" << std::endl;
    return mDefault->gains (fCell); 
  }
  return mGains->getValues (poolDetId (fCell));
}
  // gain width
const float* HcalDbServicePool::gainErrors (const cms::HcalDetId& fCell) const {
  if (!mGainWidths) {
    std::cerr << "HcalDbServicePool::gainErrors-> Gain widths are not defined. Use defaults" << std::endl;
    return mDefault->gainErrors (fCell);
  }
  return mGainWidths->getValues (poolDetId (fCell));
}
// offset for the (cell,capId,range)
const float* HcalDbServicePool::offsets (const cms::HcalDetId& fCell) const {
  std::cerr << " HcalDbServicePool::offsets-> offsets are not defined. Use defaults" << std::endl;
  return mDefault->offsets (fCell);
}
// slope for the (cell,capId,range)
const float* HcalDbServicePool::slopes (const cms::HcalDetId& fCell) const {
  std::cerr << " HcalDbServicePool::slopes-> slopes are not defined. Use defaults" << std::endl;
  return mDefault->slopes (fCell);
  return 0; 
}

HcalDbServiceBase* HcalDbServicePool::clone () const {
  return (HcalDbServiceBase*) new HcalDbServicePool ();
}

