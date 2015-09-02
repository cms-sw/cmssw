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

  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *unSortedJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *preGtEtaJets = new std::vector<l1t::Jet>();

  HICaloRingSubtraction(regions, subRegions, regionPUSParams, regionPUSType);
  //TwoByTwoFinder(subRegions, unSortedJets);
  slidingWindowJetFinder(0, subRegions, unSortedJets);
  SortJets(unSortedJets, preGtEtaJets);
  JetToGtEtaScales(params_, preGtEtaJets, preGtJets);
  JetToGtPtScales(params_, preGtJets, jets);

  delete subRegions;
  delete unSortedJets;
  delete preGtEtaJets;

  const bool verbose = true;
  const bool hex = false;
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
      if(!hex)
      {
	unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
	cout << bitset<15>(packed).to_string() << endl;
      } else {
	uint32_t output = itJet->hwPt() + (itJet->hwEta() << 6) + (itJet->hwPhi() << 10);
	std::cout << std::hex << std::setw(4) << std::setfill('0') << output << std::endl;
      }
      if(cJets == 4) break;
    }

    printf("Jets Forward\n");
    //printf("pt\teta\tphi\n");
    for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
	itJet != jets->end(); ++itJet){
      if((itJet->hwQual() & 2) != 2) continue;
      fJets++;
      if(!hex)
      {
	unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
	cout << bitset<15>(packed).to_string() << endl;
      } else {
	uint32_t output = itJet->hwPt() + (itJet->hwEta() << 6) + (itJet->hwPhi() << 10);
	std::cout << std::hex << std::setw(4) << std::setfill('0') << output << std::endl;
      }

      if(fJets == 4) break;
    }
  }

}
