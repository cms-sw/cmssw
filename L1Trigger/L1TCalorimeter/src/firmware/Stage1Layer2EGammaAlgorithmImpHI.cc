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


Stage1Layer2EGammaAlgorithmImpHI::Stage1Layer2EGammaAlgorithmImpHI(CaloParamsStage1* params) : params_(params) {};

Stage1Layer2EGammaAlgorithmImpHI::~Stage1Layer2EGammaAlgorithmImpHI(){};



void l1t::Stage1Layer2EGammaAlgorithmImpHI::processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
							 const std::vector<l1t::CaloRegion> & regions,
							 const std::vector<l1t::Jet> * jets,
							 std::vector<l1t::EGamma>* egammas) {
  std::vector<l1t::EGamma> *preSortEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *preGtEGammas = new std::vector<l1t::EGamma>();

  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = ((1-egCand->hwIso())*4 + egCand->hwQual()) ;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    int isoFlag = 0;
    bool isinBarrel = (eg_eta>=7 && eg_eta<=14);

    isoFlag = !isinBarrel;
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, isoFlag);
    preSortEGammas->push_back(theEG);
  }

  SortEGammas(preSortEGammas, preGtEGammas);
  EGammaToGtScales(params_, preGtEGammas, egammas);

  const bool verbose = false;
  if(verbose)
  {
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

  delete preSortEGammas;
  delete preGtEGammas;
}
