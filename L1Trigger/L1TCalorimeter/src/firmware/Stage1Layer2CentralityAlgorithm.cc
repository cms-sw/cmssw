///
/// \class l1t::Stage1Layer2CentralityAlgorithm
///
/// \authors: Gian Michele Innocenti
///           R. Alex Barbieri
///
/// Description: Centrality Algorithm HI

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

l1t::Stage1Layer2CentralityAlgorithm::Stage1Layer2CentralityAlgorithm(CaloParamsHelper* params)
  : params_(params)
{}


l1t::Stage1Layer2CentralityAlgorithm::~Stage1Layer2CentralityAlgorithm()
{}


void l1t::Stage1Layer2CentralityAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							const std::vector<l1t::Tau> * taus,
							l1t::CaloSpare * spare) {

  // This is no really two algorithms, the first is the HI centrality algorithm
  // while the second is an alternative MB trigger.

  // Begin Centrality Trigger //
  int etaMask = params_->centralityRegionMask();
  int sumET = 0;
  int regionET=0;

  for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {

    int etaVal = region->hwEta();
    if (etaVal > 3 && etaVal < 18) continue; // never consider central regions, independent of mask
    if((etaMask & (1<<etaVal))>>etaVal) continue;

    regionET=region->hwPt();
    sumET +=regionET;
  }

  // The LUT format is pretty funky.
  int LUT_under[8];
  int LUT_nominal[8];
  int LUT_over[8];
  for(int i = 0; i < 8; ++i)
  {
    LUT_nominal[i] = params_->centralityLUT()->data(i);
  }
  LUT_under[0] = LUT_nominal[0];
  LUT_over[0] = LUT_nominal[0];
  for(int i = 8; i < 22; ++i)
  {
    int j=i-8;
    if(j%2 == 0){
      LUT_under[j/2+1] = params_->centralityLUT()->data(i);
    } else {
      LUT_over[j/2+1] = params_->centralityLUT()->data(i);
    }
  }

  int regularResult = 0;
  int underlapResult = 0;
  int overlapResult = 0;

  for(int i = 0; i < 8; ++i)
  {
    if(sumET > LUT_nominal[i])
      regularResult = i;
    if(sumET > LUT_under[i])
      underlapResult = i;
    if(sumET >= LUT_over[i]) // logical expression in firmware is constructed slightly differently, but this is equivalent
      overlapResult = i;
  }

  int alternateResult = 0;
  if(underlapResult > regularResult) {
    alternateResult = underlapResult;
  } else if(overlapResult < regularResult) {
    alternateResult = overlapResult;
  } else {
    alternateResult = regularResult;
  }

  //paranoia
  if(regularResult > 0x7) regularResult = 0x7;
  if(alternateResult > 0x7) alternateResult = 0x7;

  spare->SetRing(0, regularResult);
  spare->SetRing(1, alternateResult);
  // End Centrality Trigger //

  // Begin MB Trigger //
  std::vector<int> thresholds = params_->minimumBiasThresholds();
  int numOverThresh[4] = {0};
  if(thresholds.size() >= 4){ // guard against malformed/old GT
    for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
      if(region->hwEta() < 4) {
	if(region->hwPt() >= thresholds.at(0))
	  numOverThresh[0]++;
	if(region->hwPt() >= thresholds.at(2))
	  numOverThresh[2]++;
      }
      if(region->hwEta() > 17) {
	if(region->hwPt() >= thresholds.at(1))
	  numOverThresh[1]++;
	if(region->hwPt() >= thresholds.at(3))
	  numOverThresh[3]++;
      }
    }
  }

  int bits[6];
  bits[0] = ((numOverThresh[0] > 0) && (numOverThresh[1] > 0));
  bits[1] = ((numOverThresh[0] > 0) || (numOverThresh[1] > 0));
  bits[2] = ((numOverThresh[2] > 0) && (numOverThresh[3] > 0));
  bits[3] = ((numOverThresh[2] > 0) || (numOverThresh[3] > 0));
  bits[4] = ((numOverThresh[0] > 1) && (numOverThresh[1] > 1));
  bits[5] = ((numOverThresh[2] > 1) && (numOverThresh[3] > 1));

  spare->SetRing(2, (bits[2]<<2) + (bits[1]<<1) + bits[0]);
  spare->SetRing(3, (bits[5]<<2) + (bits[4]<<1) + bits[3]);
  // End MB Trigger //

  const bool verbose = false;
  const bool hex = true;
  if(verbose)
  {
    if(!hex)
    {
      std::cout << "HF Ring Sums (Centrality)" << std::endl;
      std::cout << bitset<12>(spare->hwPt()).to_string() << std::endl;
    } else {
      std::cout << "Centrality" << std::endl;
      std::cout << std::hex << spare->hwPt() << std::endl;
      // std::cout << std::hex << spare->GetRing(0) << " "
      // 		<< spare->GetRing(1) << " "
      // 		<< bits[0] << " " << bits[1] << " "
      // 		<< bits[2] << " " << bits[3] << " "
      // 		<< bits[4] << " " << bits[5] << std::endl;
    }
  }

}
