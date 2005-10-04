
//
// F.Ratnikov (UMd), Jul. 19, 2005
//

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"


namespace {
  const float binMin [33] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			      9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62, 67};
  int range (int adc) {
    return (adc >> 5) & 3;
  }
};

HcalDbServiceHardcode::HcalDbServiceHardcode () {}
HcalDbServiceHardcode::~HcalDbServiceHardcode () {}


const char* HcalDbServiceHardcode::name () const {return "HcalDbServiceHardcode";}

// basic conversion function for single range (0<=count<32)
double HcalDbServiceHardcode::adcShape (int fCount) const {
  return 0.5 * (binMin[fCount] + binMin[fCount+1]);
}
// bin size for the QIE conversion
double HcalDbServiceHardcode::adcShapeBin (int fCount) const {
  return binMin[fCount+1] - binMin[fCount];
}
// pedestal  
const float* HcalDbServiceHardcode::pedestals (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) pedestal [i] = 0.75; // 750MeV
  return pedestal;
}
// pedestal width
const float* HcalDbServiceHardcode::pedestalErrors (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) pError [i] = 0.; // none
  return pError;
}
// gain
const float* HcalDbServiceHardcode::gains (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) gain [i] = fCell.subdet () == cms::HcalForward ? 0.150 : 0.177; // GeV/fC
  return gain;
}
// gain width
const float* HcalDbServiceHardcode::gainErrors (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) gError [i] = 0.; // none
  return gError;
}
// offset for the (cell,capId,range)
const float* HcalDbServiceHardcode::offsets (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) {
    int irange = 4;
    while (irange <= 0) {
      offset [HcalDbServiceBase::index (i, irange)] = 0.; // none
    }
  }
  return offset;
}
// slope for the (cell,capId,range)
const float* HcalDbServiceHardcode::slopes (const cms::HcalDetId& fCell) const {
  int i = 4;
  while (--i >= 0) {
    int irange = 4;
    while (irange <= 0) {
      slope [HcalDbServiceBase::index (i, irange)] = fCell.subdet () == cms::HcalForward ? 2.6 : 1.;
    }
  }
  return slope;
}

HcalDbServiceBase* HcalDbServiceHardcode::clone () const {
  return (HcalDbServiceBase*) new HcalDbServiceHardcode ();
}
