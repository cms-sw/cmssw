// PUSubtractionMethods.h
// Authors: Alex Barbieri
//          Kalanand Mishra, Fermilab
//          Inga Bucinskaite, UIC
//
// This file should contain the different algorithms used for PU subtraction.

#ifndef PUSUBTRACTIONMETHODS_H
#define PUSUBTRACTIONMETHODS_H

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
//#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsStage1.h"

#include <vector>

namespace l1t {

  void HICaloRingSubtraction(const std::vector<l1t::CaloRegion> & regions,
			     std::vector<l1t::CaloRegion> *subRegions,
			     std::vector<double> regionPUSparams,
			     std::string regionPUSType);

  void simpleHWSubtraction(const std::vector<l1t::CaloRegion> & regions,
			   std::vector<l1t::CaloRegion> *subRegions);

  void RegionCorrection(const std::vector<l1t::CaloRegion> & regions,
			std::vector<l1t::CaloRegion> *subRegions,
			std::vector<double> regionPUSparams,
			std::string regionPUSType);


}

#endif
