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

// Taken from UCT code. Might not be appropriate. Refers to legacy L1 objects.
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//#include "DataFormats/Candidate/interface/LeafCandidate.h"
//#include <stdio.h>

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpHI::Stage1Layer2JetAlgorithmImpHI(CaloParams* params) : params_(params)
{
  double jetScale=params_->jetScale();
  jetSeedThreshold= floor( params_->jetSeedThreshold()/jetScale + 0.5);
}
//: regionLSB_(0.5) {}

Stage1Layer2JetAlgorithmImpHI::~Stage1Layer2JetAlgorithmImpHI(){};

void Stage1Layer2JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
					       std::vector<l1t::Jet> * jets){

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  HICaloRingSubtraction(regions, subRegions);
  slidingWindowJetFinder(jetSeedThreshold, subRegions, jets);

  delete subRegions;
}
