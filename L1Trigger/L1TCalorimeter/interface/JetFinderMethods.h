// JetFinderMethods.h
// Author: Alex Barbieri
//
// This file should contain the different algorithms used to find jets.
// Currently the standard is the sliding window method, used by both
// HI and PP.

#ifndef JETFINDERMETHODS_H
#define JETFINDERMETHODS_H

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include <vector>

namespace l1t {

  int deltaGctPhi(const CaloRegion & region, const CaloRegion & neighbor);
  void slidingWindowJetFinder(const int, const std::vector<l1t::CaloRegion> * regions,
			      std::vector<l1t::Jet> * uncalibjets);
  void TwelveByTwelveFinder(const int, const std::vector<l1t::CaloRegion> * regions,
			      std::vector<l1t::Jet> * uncalibjets);
  void passThroughJets(const std::vector<l1t::CaloRegion> * regions,
		       std::vector<l1t::Jet> * uncalibjets);
  void TwoByTwoFinder(const int, const int, const std::vector<l1t::CaloRegion> * regions,
		      std::vector<l1t::Jet> * uncalibjets);
}

#endif
