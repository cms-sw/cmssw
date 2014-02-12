// PUSubtractionMethods.h
// Author: Alex Barbieri
//
// This file should contain the different algorithms used for PU subtraction.

#ifndef PUSUBTRACTIONMETHODS_H
#define PUSUBTRACTIONMETHODS_H

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include <vector>

namespace l1t {

  void HICaloRingSubtraction(const std::vector<l1t::CaloRegion> & regions,
			     std::vector<l1t::CaloRegion> *subRegions);
}

#endif
