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
#include "L1Trigger/L1TCalorimeter/interface/JetCalibrationMethods.h"

// Taken from UCT code. Might not be appropriate. Refers to legacy L1 objects.
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//#include "DataFormats/Candidate/interface/LeafCandidate.h"

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpPP::Stage1Layer2JetAlgorithmImpPP(CaloParams* params) : params_(params)
{
  jetLsb=params_->jetLsb();
  jetSeedThreshold= floor( params_->jetSeedThreshold()/jetLsb + 0.5);
  regionPUSType = params_->regionPUSType();
  regionPUSParams = params_->regionPUSParams();
  jetCalibrationType = params_->jetCalibrationType();
  jetCalibrationParams = params_->jetCalibrationParams();
}
//: regionLSB_(0.5) {}

Stage1Layer2JetAlgorithmImpPP::~Stage1Layer2JetAlgorithmImpPP(){};


void Stage1Layer2JetAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets){


  std::vector<l1t::CaloRegion> * subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> * uncalibjets = new std::vector<l1t::Jet>();

  
  //Region Correction will return uncorrected subregions 
  //if regionPUSType is set to None in the config
  RegionCorrection(regions, EMCands, subRegions, regionPUSParams, regionPUSType);
  
  
  slidingWindowJetFinder(jetSeedThreshold, subRegions, uncalibjets);

  //will return jets with no response corrections
  //if jetCalibrationType is set to None in the config
  JetCalibration(uncalibjets, jetCalibrationParams, jets, jetCalibrationType, jetLsb);


  delete subRegions;
  delete uncalibjets;

  // std::vector<l1t::CaloRegion>::const_iterator incell;
  // for (incell = regions.begin(); incell != regions.end(); ++incell){
  //   //do nothing for now
  // }

}

