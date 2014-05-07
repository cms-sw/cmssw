// JetCalibrationMethods.h
// Authors: Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used for Jet Calibration

#ifndef JETCALIBRATIONMETHODS_H
#define JETCALIBRATIONMETHODS_H

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include <vector>

namespace l1t {
  
  void JetCalibration1(std::vector<l1t::Jet> * uncalibjets,
		       std::vector<double> jetSF,
		       std::vector<l1t::Jet> * jets,
		       bool applyJetCalibration,
		       double jetLSB); 
  
}

#endif
