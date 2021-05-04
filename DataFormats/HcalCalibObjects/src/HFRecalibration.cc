///////////////////////////////////////////////////////////////////////////////
// File: HFRecalibration.cc
// Description: simple helper class containing parameterized
//              function for HF damade recovery for Upgrade studies
//              evaluated using SimG4CMS/Calo/ HFDarkening
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/HcalCalibObjects/interface/HFRecalibration.h"

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

HFRecalibration::HFRecalibration(const edm::ParameterSet& pset) {
  //HFParsAB[Depth=0/1][A=0/B=1]
  HFParsAB[0][0] = pset.getParameter<vecOfDoubles>("HFdepthOneParameterA");
  HFParsAB[0][1] = pset.getParameter<vecOfDoubles>("HFdepthOneParameterB");
  HFParsAB[1][0] = pset.getParameter<vecOfDoubles>("HFdepthTwoParameterA");
  HFParsAB[1][1] = pset.getParameter<vecOfDoubles>("HFdepthTwoParameterB");
}

HFRecalibration::~HFRecalibration() {}

double HFRecalibration::getCorr(int ieta, int depth, double lumi) {
  // parameterizations provided by James Wetzel

  ieta = (abs(ieta) - loweriEtaBin);

  if (ieta < 0 || ieta > 11 || depth < 1 || depth > 2) {
    return 1.0;
  }

  switch (depth) {
    case 1:
      reCalFactor = (1 + HFParsAB[0][0][ieta] * sqrt(lumi) + HFParsAB[0][1][ieta] * lumi);
      break;
    case 2:
      reCalFactor = (1 + HFParsAB[1][0][ieta] * sqrt(lumi) + HFParsAB[1][1][ieta] * lumi);
  }

  return reCalFactor;
}
