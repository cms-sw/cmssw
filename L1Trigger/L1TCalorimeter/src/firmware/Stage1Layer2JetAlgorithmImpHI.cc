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

Stage1Layer2JetAlgorithmImpHI::Stage1Layer2JetAlgorithmImpHI(CaloParamsStage1* params) : params_(params) { };

Stage1Layer2JetAlgorithmImpHI::~Stage1Layer2JetAlgorithmImpHI(){};

void Stage1Layer2JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets,
						 std::vector<l1t::Jet> * preGtJets ){

  double towerLsb = params_->towerLsbSum();
  int jetSeedThreshold= floor( params_->jetSeedThreshold()/towerLsb + 0.5);

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *unSortedJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *preGtEtaJets = new std::vector<l1t::Jet>();

  HICaloRingSubtraction(regions, subRegions);
  slidingWindowJetFinder(jetSeedThreshold, subRegions, unSortedJets);
  SortJets(unSortedJets, preGtEtaJets);
  JetToGtEtaScales(params_, preGtEtaJets, preGtJets);
  JetToGtPtScales(params_, preGtJets, jets);

  delete subRegions;
  delete unSortedJets;
  delete preGtEtaJets;
  //delete preGtJets;

  //the jets should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  // auto comp = [&](l1t::Jet i, l1t::Jet j)-> bool {
  //   return (i.hwPt() < j.hwPt() );
  // };

  // std::sort(jets->begin(), jets->end(), comp);
  // std::reverse(jets->begin(), jets->end());

}
