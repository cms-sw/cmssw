///////////////////////////////////////////////////////////////////////////////
// File: HFRecalibration.cc
// Description: simple helper class containing parameterized 
//              function for HF damade recovery for Upgrade studies  
//              evaluated using SimG4CMS/Calo/ HFDarkening   
///////////////////////////////////////////////////////////////////////////////

#include "HFRecalibration.h"

HFRecalibration::HFRecalibration() { }
HFRecalibration::~HFRecalibration() { }

double HFRecalibration::getCorr(int ieta, int depth, double lumi) {
  return 1.0;
}
