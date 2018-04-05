// JetCalibrationMethods.h
// Authors: Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used for Jet Calibration

#ifndef JETCALIBRATIONMETHODS_H
#define JETCALIBRATIONMETHODS_H

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include <vector>

namespace l1t {

  void JetCalibration(std::vector<l1t::Jet> * uncalibjets,
		      std::vector<double> jetCalibrationParams,
		      std::vector<l1t::Jet> * jets,
		      std::string jetCalibrationType,
		      double jetLSB);

}

#endif
