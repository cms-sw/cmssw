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

Stage1Layer2JetAlgorithmImpSimpleHW::Stage1Layer2JetAlgorithmImpSimpleHW(CaloParamsHelper* params) : params_(params)
{
}

Stage1Layer2JetAlgorithmImpSimpleHW::~Stage1Layer2JetAlgorithmImpSimpleHW(){};

void Stage1Layer2JetAlgorithmImpSimpleHW::processEvent(const std::vector<l1t::CaloRegion> & regions,
						       const std::vector<l1t::CaloEmCand> & EMCands,
						       std::vector<l1t::Jet> * jets,
						       std::vector<l1t::Jet> * debugJets){

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *preGtEtaJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *calibratedRankedJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *sortedJets = new std::vector<l1t::Jet>();

  double towerLsb = params_->towerLsbSum();
  int jetSeedThreshold = floor( params_->jetSeedThreshold()/towerLsb + 0.5);

  RegionCorrection(regions, subRegions, params_);

  slidingWindowJetFinder(jetSeedThreshold, subRegions, preGtEtaJets);

  calibrateAndRankJets(params_, preGtEtaJets, calibratedRankedJets);

  SortJets(calibratedRankedJets, sortedJets);

  JetToGtEtaScales(params_, sortedJets, jets);
  JetToGtEtaScales(params_, preGtEtaJets, debugJets);
  //JetToGtPtScales(params_, preGtJets, jets);

  delete subRegions;
  delete preGtEtaJets;
  delete calibratedRankedJets;
  delete sortedJets;
}
