
//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#include <iostream>

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
double HcalDbServiceHardcode::pedestal (const cms::HcalDetId& fCell, int fCapId) const {
  // return 0.75; // 750MeV
  double result = fCell.subdet () == cms::HcalForward ? 4.*2.6 : 0.75; // GeV. Corresponds to Jeremy's constant 
  //  std::cout << "HcalDbServiceHardcode::pedestal-> " << fCell << " / " << fCapId << ": " << result << std::endl;
  return result;
}
  // pedestal width
double HcalDbServiceHardcode::pedestalError (const cms::HcalDetId& fCell, int fCapId) const {
  return 0.; // none
}
  // gain
double HcalDbServiceHardcode::gain (const cms::HcalDetId& fCell, int fCapId) const {
  //  return fCell.subdet () == cms::HcalForward ? 0.150 : 0.177; // GeV/fC 
  double result = fCell.subdet () == cms::HcalForward ? 0.2 / 2.6 : 0.177; // GeV/fC Corresponds to Jeremy's constant 
  //  std::cout << "HcalDbServiceHardcode::gain-> " << fCell << " / " << fCapId << ": " << result << std::endl;
  return result;
}
  // gain width
double HcalDbServiceHardcode::gainError (const cms::HcalDetId& fCell, int fCapId) const {
  return 0.; // none
}
// offset for the (cell,capId,range)
double HcalDbServiceHardcode::offset (const cms::HcalDetId& fCell, int fCapId, int fRange) const {
  return 0;
}
// slope for the (cell,capId,range)
double HcalDbServiceHardcode::slope (const cms::HcalDetId& fCell, int fCapId, int fRange) const {
  return fCell.subdet () == cms::HcalForward ? 2.6 : 1.; 
}

HcalDbService* HcalDbServiceHardcode::clone () const {
  return (HcalDbService*) new HcalDbServiceHardcode ();
}

EVENTSETUP_DATA_REG(HcalDbServiceHardcode);
