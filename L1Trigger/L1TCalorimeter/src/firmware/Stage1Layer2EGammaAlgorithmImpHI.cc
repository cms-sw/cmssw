///step03
/// \class l1t::Stage1Layer2EGammaAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EGammaAlgorithmImp.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

#include <bitset>

using namespace std;
using namespace l1t;


Stage1Layer2EGammaAlgorithmImpHI::Stage1Layer2EGammaAlgorithmImpHI(CaloParamsHelper* params) : params_(params) {};

Stage1Layer2EGammaAlgorithmImpHI::~Stage1Layer2EGammaAlgorithmImpHI(){};

void verboseDumpEGammas(const std::vector<l1t::EGamma> &egs);

void l1t::Stage1Layer2EGammaAlgorithmImpHI::processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
							 const std::vector<l1t::CaloRegion> & regions,
							 const std::vector<l1t::Jet> * jets,
							 std::vector<l1t::EGamma>* egammas) {
  int egEtaCut = params_->egEtaCut();

  std::vector<l1t::EGamma> *preSortEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *preGtEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *dumpEGammas = new std::vector<l1t::EGamma>();

  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = (egCand->hwIso()*4 + egCand->hwQual()) ;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    int isoFlag = 0;
    bool isinBarrel = false;
    if((egEtaCut & (1<<eg_eta))>>eg_eta) {
      isinBarrel = true;
    }

    isoFlag = isinBarrel;
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, isoFlag);
    preSortEGammas->push_back(theEG);
  }

  //EGammaToGtScales(params_, preSortEGammas, dumpEGammas);
  //verboseDumpEGammas(*dumpEGammas);

  SortEGammas(preSortEGammas, preGtEGammas);
  EGammaToGtScales(params_, preGtEGammas, egammas);

  const bool verbose = false;
  const bool hex = true;
  if(verbose)
  {
    if(hex)
    {
      std::cout << "EGammas" << std::endl;
      l1t::EGamma aegammas[8];
      for(std::vector<l1t::EGamma>::const_iterator itEgamma = egammas->begin();
	  itEgamma != egammas->end(); ++itEgamma){
	aegammas[itEgamma - egammas->begin()] = *itEgamma;
      }
      //std::cout << "Egammas (hex)" << std::endl;
      std::cout << std::hex << pack16bits(aegammas[0].hwPt(), aegammas[0].hwEta(), aegammas[0].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[1].hwPt(), aegammas[1].hwEta(), aegammas[1].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[4].hwPt(), aegammas[4].hwEta(), aegammas[4].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[5].hwPt(), aegammas[5].hwEta(), aegammas[5].hwPhi());
      std::cout << std::endl;
      std::cout << std::hex << pack16bits(aegammas[2].hwPt(), aegammas[2].hwEta(), aegammas[2].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[3].hwPt(), aegammas[3].hwEta(), aegammas[3].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[6].hwPt(), aegammas[6].hwEta(), aegammas[6].hwPhi());
      std::cout << " ";
      std::cout << std::hex << pack16bits(aegammas[7].hwPt(), aegammas[7].hwEta(), aegammas[7].hwPhi());
      std::cout << std::endl;
    } else {
      int cEGammas = 0;
      int fEGammas = 0;
      printf("EGammas Isolated\n");
      for(std::vector<l1t::EGamma>::const_iterator itEGamma = egammas->begin();
	  itEGamma != egammas->end(); ++itEGamma){
	if(itEGamma->hwIso() != 1) continue;
	cEGammas++;
	unsigned int packed = pack15bits(itEGamma->hwPt(), itEGamma->hwEta(), itEGamma->hwPhi());
	cout << bitset<15>(packed).to_string() << endl;
	if(cEGammas == 4) break;
      }

      printf("EGammas Non-isolated\n");
      //printf("pt\teta\tphi\n");
      for(std::vector<l1t::EGamma>::const_iterator itEGamma = egammas->begin();
	  itEGamma != egammas->end(); ++itEGamma){
	if(itEGamma->hwIso() != 0) continue;
	fEGammas++;
	unsigned int packed = pack15bits(itEGamma->hwPt(), itEGamma->hwEta(), itEGamma->hwPhi());
	cout << bitset<15>(packed).to_string() << endl;
	if(fEGammas == 4) break;
      }
    }
  }

  delete preSortEGammas;
  delete preGtEGammas;
  delete dumpEGammas;
}

void verboseDumpEGammas(const std::vector<l1t::EGamma> &jets)
{
  // int fwPhi[18] = {	4,
  // 		   	3 ,
  // 		   	2  ,
  // 		   	1   ,
  // 		   	0   ,
  // 		   	17  ,
  // 		   	16  ,
  // 		   	15  ,
  // 		   	14  ,
  // 		   	13  ,
  // 		  	12  ,
  // 		  	11  ,
  // 		  	10  ,
  // 		  	9   ,
  // 		  	8   ,
  // 		  	7   ,
  // 		  	6   ,
  // 		  	5   };

  // int fwEta[22] = {0,
  // 		   1,
  // 		   2,
  // 		   3,
  //   		   0,
  // 		   1,
  // 		   2,
  // 		   3,
  // 		   4,
  // 		   5,
  // 		   6,
  // 		   0,
  // 		   1,
  // 		   2,
  // 		   3,
  // 		   4,
  // 		   5,
  // 		   6,
  // 		   0,
  // 		   1,
  // 		   2,
  //   		   3};


  std::cout << "pt eta phi" << std::endl;
  for(std::vector<l1t::EGamma>::const_iterator itEGamma = jets.begin();
	  itEGamma != jets.end(); ++itEGamma){

    //std::cout << itEGamma->hwPt() << " ";
    //std::cout << fwEta[itEGamma->hwEta()] << " " ;
    //std::cout << fwPhi[itEGamma->hwPhi()] << " ";
    //std::cout << itEGamma->hwEta() << " " ;
    //std::cout << itEGamma->hwPhi() << std::endl;
    //bool sign = (itEGamma->hwEta() < 11);
    //std::cout << sign << std::endl;
    std::cout << std::hex << pack16bitsEgammaSpecial(itEGamma->hwPt(), itEGamma->hwEta(), itEGamma->hwPhi()) << std::endl;


  }
}
