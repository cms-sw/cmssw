///
/// \class l1t::Stage1Layer2JetAlgorithmImpSimpleHW
///
///
/// \author: R. Alex Barbieri MIT
///

// This is a simple algorithm for use in comparing with early versions of the Stage1 firmware

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2JetAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"

#include <bitset>
#include <iostream>

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpSimpleHW::Stage1Layer2JetAlgorithmImpSimpleHW(CaloParamsHelper const* params)
    : params_(params) {}

void Stage1Layer2JetAlgorithmImpSimpleHW::processEvent(const std::vector<l1t::CaloRegion>& regions,
                                                       const std::vector<l1t::CaloEmCand>& EMCands,
                                                       std::vector<l1t::Jet>* jets,
                                                       std::vector<l1t::Jet>* debugJets) {
  std::vector<l1t::CaloRegion> subRegions;
  std::vector<l1t::Jet> preGtEtaJets;
  std::vector<l1t::Jet> calibratedRankedJets;
  std::vector<l1t::Jet> sortedJets;

  double towerLsb = params_->towerLsbSum();
  int jetSeedThreshold = floor(params_->jetSeedThreshold() / towerLsb + 0.5);

  RegionCorrection(regions, &subRegions, params_);

  slidingWindowJetFinder(jetSeedThreshold, &subRegions, &preGtEtaJets);

  calibrateAndRankJets(params_, &preGtEtaJets, &calibratedRankedJets);

  SortJets(&calibratedRankedJets, &sortedJets);

  JetToGtEtaScales(params_, &sortedJets, jets);
  JetToGtEtaScales(params_, &preGtEtaJets, debugJets);
  //JetToGtPtScales(params_, preGtJets, jets);
}
