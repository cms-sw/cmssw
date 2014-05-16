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
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpPP::Stage1Layer2JetAlgorithmImpPP(CaloParamsStage1* params) : params_(params)
{
  jetLsb=params_->jetLsb();
  jetSeedThreshold= floor( params_->jetSeedThreshold()/jetLsb + 0.5);
  regionPUSType = params_->regionPUSType();
  regionPUSParams = params_->regionPUSParams();
  jetCalibrationType = params_->jetCalibrationType();
  jetCalibrationParams = params_->jetCalibrationParams();
}

Stage1Layer2JetAlgorithmImpPP::~Stage1Layer2JetAlgorithmImpPP(){};


void Stage1Layer2JetAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets){


  std::vector<l1t::CaloRegion> * subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> * uncalibjets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> * preGtJets = new std::vector<l1t::Jet>();


  //Region Correction will return uncorrected subregions
  //if regionPUSType is set to None in the config
  RegionCorrection(regions, EMCands, subRegions, regionPUSParams, regionPUSType);


  slidingWindowJetFinder(jetSeedThreshold, subRegions, uncalibjets);

  //will return jets with no response corrections
  //if jetCalibrationType is set to None in the config
  JetCalibration(uncalibjets, jetCalibrationParams, preGtJets, jetCalibrationType, jetLsb);

  // takes input jets (using region scales/eta) and outputs jets using Gt scales/eta
  JetToGtScales(params_, preGtJets, jets);

  delete subRegions;
  delete uncalibjets;
  delete preGtJets;

  //the jets should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  auto comp = [&](l1t::Jet i, l1t::Jet j)-> bool {
    return (i.hwPt() < j.hwPt() );
  };

  std::sort(jets->begin(), jets->end(), comp);
  std::reverse(jets->begin(), jets->end());
}
