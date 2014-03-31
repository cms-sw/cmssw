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

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpPP::Stage1Layer2JetAlgorithmImpPP(CaloParams* params) : params_(params)
{
  double jetScale=params_->jetScale();
  jetSeedThreshold= floor( params_->jetSeedThreshold()/jetScale + 0.5);

  
}
//: regionLSB_(0.5) {}

Stage1Layer2JetAlgorithmImpPP::~Stage1Layer2JetAlgorithmImpPP(){};

void puSubtractionPP(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> * subRegions);


void Stage1Layer2JetAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets){


  std::vector<l1t::CaloRegion> * subRegions = new std::vector<l1t::CaloRegion>();
  bool Correct=true;
  // bool Correct=false;
  if (Correct){
    RegionCorrection(regions, EMCands, subRegions);
  }else{
    // subRegions = *regions;
    for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
      CaloRegion newSubRegion= *region;
      subRegions->push_back(newSubRegion);
    }
    // puSubtractionPP(regions, subRegions);
  }

  
  slidingWindowJetFinder(jetSeedThreshold, subRegions, jets);
  delete subRegions;

  // std::vector<l1t::CaloRegion>::const_iterator incell;
  // for (incell = regions.begin(); incell != regions.end(); ++incell){
  //   //do nothing for now
  // }

}

// NB PU is not in the physical scale!!  Needs to be multiplied by regionLSB
void puSubtractionPP(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> * subRegions)
{
  // just copy regions into subRegions for now so jetfinder has something to work with

  for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
    CaloRegion newSubRegion= *region;
    subRegions->push_back(newSubRegion);
    }
}
