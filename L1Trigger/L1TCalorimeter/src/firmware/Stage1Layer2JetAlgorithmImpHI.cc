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

Stage1Layer2JetAlgorithmImpHI::Stage1Layer2JetAlgorithmImpHI(CaloParamsHelper* params) : params_(params) { };

Stage1Layer2JetAlgorithmImpHI::~Stage1Layer2JetAlgorithmImpHI(){};

void verboseDumpJets(const std::vector<l1t::Jet> &jets);

void Stage1Layer2JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets,
						 std::vector<l1t::Jet> * preGtJets ){

  //std::vector<double> regionPUSParams = params_->regionPUSParams();
  int jetThreshold = params_->jetSeedThreshold();

  unsigned int etaMask = params_->jetRegionMask();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *unSortedJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *preGtEtaJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *preRankJets = new std::vector<l1t::Jet>();

  HICaloRingSubtraction(regions, subRegions, params_);
  TwoByTwoFinder(jetThreshold, etaMask, subRegions, preRankJets);
  //slidingWindowJetFinder(0, subRegions, unSortedJets);
  JetToGtPtScales(params_, preRankJets, unSortedJets);
  //verboseDumpJets(*unSortedJets);
  SortJets(unSortedJets, preGtEtaJets);
  JetToGtEtaScales(params_, preGtEtaJets, preGtJets);
  JetToGtEtaScales(params_, preGtEtaJets, jets);
  //JetToGtPtScales(params_, preGtJets, jets);

  delete subRegions;
  delete unSortedJets;
  delete preGtEtaJets;

  const bool verbose = false;
  const bool hex = true;
  if(verbose)
  {
    if(!hex)
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
    } else {
      std::cout << "Jets" << std::endl;
      l1t::Jet ajets[8];
      for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
	  itJet != jets->end(); ++itJet){
	ajets[itJet - jets->begin()] = *itJet;
      }
      //std::cout << "Jets (hex)" << std::endl;
      std::cout << std::hex << pack16bits(ajets[0].hwPt(), ajets[0].hwEta(), ajets[0].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[1].hwPt(), ajets[1].hwEta(), ajets[1].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[4].hwPt(), ajets[4].hwEta(), ajets[4].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[5].hwPt(), ajets[5].hwEta(), ajets[5].hwPhi());
      std::cout << std::endl;
      std::cout << std::hex << pack16bits(ajets[2].hwPt(), ajets[2].hwEta(), ajets[2].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[3].hwPt(), ajets[3].hwEta(), ajets[3].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[6].hwPt(), ajets[6].hwEta(), ajets[6].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(ajets[7].hwPt(), ajets[7].hwEta(), ajets[7].hwPhi());
      std::cout << std::endl;
    }
  }
}

void verboseDumpJets(const std::vector<l1t::Jet> &jets)
{
  int fwPhi[18] = {	4,
		   	3 ,
		   	2  ,
		   	1   ,
		   	0   ,
		   	17  ,
		   	16  ,
		   	15  ,
		   	14  ,
		   	13  ,
		  	12  ,
		  	11  ,
		  	10  ,
		  	9   ,
		  	8   ,
		  	7   ,
		  	6   ,
		  	5   };

  int fwEta[22] = {0,
		   1,
		   2,
		   3,
    		   0,
		   1,
		   2,
		   3,
		   4,
		   5,
		   6,
		   0,
		   1,
		   2,
		   3,
		   4,
		   5,
		   6,
		   0,
		   1,
		   2,
    		   3};


  std::cout << "pt eta phi sign" << std::endl;
  for(std::vector<l1t::Jet>::const_iterator itJet = jets.begin();
	  itJet != jets.end(); ++itJet){

    std::cout << itJet->hwPt() << " ";
    std::cout << fwEta[itJet->hwEta()] << " " ;
    std::cout << fwPhi[itJet->hwPhi()] << " ";
    bool sign = (itJet->hwEta() < 11);
    std::cout << sign << std::endl;

  }
}
