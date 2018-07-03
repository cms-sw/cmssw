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
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include <vector>

namespace l1t {
  void SortJets(std::vector<l1t::Jet> * input,
                std::vector<l1t::Jet> * output);

  void SortEGammas(std::vector<l1t::EGamma> * input,
		   std::vector<l1t::EGamma> * output);

  void SortTaus(std::vector<l1t::Tau> * input,
                std::vector<l1t::Tau> * output);

  unsigned int pack15bits(int pt, int eta, int phi);
  unsigned int pack16bits(int pt, int eta, int phi);
  unsigned int pack16bitsEgammaSpecial(int pt, int eta, int phi);
}

#endif
