///
/// \class l1t::Stage1Layer2JetAlgorithmImpHI
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implements algorithm version 1 and 2.

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2JetAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpHI::Stage1Layer2JetAlgorithmImpHI(CaloParamsHelper const* params) : params_(params){};

void Stage1Layer2JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion>& regions,
                                                 const std::vector<l1t::CaloEmCand>& EMCands,
                                                 std::vector<l1t::Jet>* jets,
                                                 std::vector<l1t::Jet>* preGtJets) {
  //std::vector<double> regionPUSParams = params_->regionPUSParams();
  int jetThreshold = params_->jetSeedThreshold();

  unsigned int etaMask = params_->jetRegionMask();

  std::vector<l1t::CaloRegion> subRegions;
  std::vector<l1t::Jet> unSortedJets;
  std::vector<l1t::Jet> preGtEtaJets;
  std::vector<l1t::Jet> preRankJets;

  HICaloRingSubtraction(regions, &subRegions, params_);
  TwoByTwoFinder(jetThreshold, etaMask, &subRegions, &preRankJets);
  //slidingWindowJetFinder(0, subRegions, unSortedJets);
  JetToGtPtScales(params_, &preRankJets, &unSortedJets);
  SortJets(&unSortedJets, &preGtEtaJets);
  JetToGtEtaScales(params_, &preGtEtaJets, preGtJets);
  JetToGtEtaScales(params_, &preGtEtaJets, jets);
  //JetToGtPtScales(params_, preGtJets, jets);
}
