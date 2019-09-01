#ifndef CONDTOOLS_HCAL_PARSEHCALDETID_H_
#define CONDTOOLS_HCAL_PARSEHCALDETID_H_

#include <string>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

//
// Construct HcalDetId from a line of text.
// Return HcalDetId with rawId of 0 of the
// text does not corrspond to a correct
// detector id specification.
//
HcalDetId parseHcalDetId(const std::string& s);

//
// Return a human-readable name for the given subdetector
//
const char* hcalSubdetectorName(HcalSubdetector subdet);

#endif  // CONDTOOLS_HCAL_PARSEHCALDETID_H_
