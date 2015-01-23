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

Stage1Layer2JetAlgorithmImpSimpleHW::Stage1Layer2JetAlgorithmImpSimpleHW(CaloParamsStage1* params) : params_(params)
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

  //simpleHWSubtraction(regions, subRegions);

  std::string regionPUSType = "PUM0"; //params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();
  RegionCorrection(regions, subRegions, regionPUSParams, regionPUSType);

  slidingWindowJetFinder(0, subRegions, preGtEtaJets);

  calibrateAndRankJets(params_, preGtEtaJets, calibratedRankedJets);

  SortJets(calibratedRankedJets, sortedJets);

  JetToGtEtaScales(params_, sortedJets, jets);
  JetToGtEtaScales(params_, preGtEtaJets, debugJets);
  //JetToGtPtScales(params_, preGtJets, jets);

  const bool verbose = true;
  if(verbose)
  {
    int cJets = 0;
    int fJets = 0;
    printf("Jets Central\n");
    //printf("pt\teta\tphi\n");
    for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
	itJet != jets->end(); ++itJet){
      if((itJet->hwQual() & 2) == 2) continue;
      cJets++;
      unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
      cout << bitset<15>(packed).to_string() << endl;
      if(cJets == 4) break;
    }

    printf("Jets Forward\n");
    //printf("pt\teta\tphi\n");
    for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
	itJet != jets->end(); ++itJet){
      if((itJet->hwQual() & 2) != 2) continue;
      fJets++;
      unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
      cout << bitset<15>(packed).to_string() << endl;
      if(fJets == 4) break;
    }
  }

  delete subRegions;
  delete preGtEtaJets;
  delete calibratedRankedJets;
  delete sortedJets;
}
