// HardwareSortingMethods.h
// Authors: R. Alex Barbieri
//          Ben Kreis
//
// This file should contain the C++ equivalents of the sorting
// algorithms used in Hardware. Most C++ methods originally written by
// Ben Kries.

#ifndef HARDWARESORTINGMETHODS_H
#define HARDWARESORTINGMETHODS_H

#include "DataFormats/L1Trigger/interface/Jet.h"
#include <vector>

namespace l1t {
  void SortJets(std::vector<l1t::Jet> * input,
                std::vector<l1t::Jet> * output);
}

#endif
