///
/// \class l1t::Stage1Layer2CentralityAlgorithm
///
/// \authors: Gian Michele Innocenti
///           R. Alex Barbieri
///
/// Description: Centrality Algorithm HI

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFBitCountAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

l1t::Stage1Layer2HFMinimumBias::Stage1Layer2HFMinimumBias(CaloParamsStage1* params)
  : params_(params)
{}


l1t::Stage1Layer2HFMinimumBias::~Stage1Layer2HFMinimumBias()
{}


void l1t::Stage1Layer2HFMinimumBias::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							l1t::CaloSpare * spare) {

  int sumBits[4] = {0,0,0,0};

  for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
    switch(region->hwEta() )
    {
    case 0: //1-
      sumBits[1] += region->hwQual();
      break;
    case 1: //2-
      sumBits[3] += region->hwQual();
      break;
    case 20: //2+
      sumBits[2] += region->hwQual();
      break;
    case 21: //1+
      sumBits[0] += region->hwQual();
      break;
    default:
      break;
    }
  }

  for(int i = 0; i < 4; i++)
  {
    if(sumBits[i] > 0x7)
      sumBits[i] = 0x7;

    spare->SetRing(i, sumBits[i]);
  }

  const bool verbose = false;
  if(verbose)
  {
    std::cout << "HF Bit Counts (HFMinimumBias)" << std::endl;
    std::cout << bitset<12>(spare->hwPt()).to_string() << std::endl;
  }
}
