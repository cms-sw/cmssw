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

Stage1Layer2EGammaAlgorithmImpHI::Stage1Layer2EGammaAlgorithmImpHI(CaloParamsHelper const* params) : params_(params){};

void l1t::Stage1Layer2EGammaAlgorithmImpHI::processEvent(const std::vector<l1t::CaloEmCand>& EMCands,
                                                         const std::vector<l1t::CaloRegion>& regions,
                                                         const std::vector<l1t::Jet>* jets,
                                                         std::vector<l1t::EGamma>* egammas) {
  int egEtaCut = params_->egEtaCut();

  std::vector<l1t::EGamma> preSortEGammas;
  std::vector<l1t::EGamma> preGtEGammas;

  for (CaloEmCandBxCollection::const_iterator egCand = EMCands.begin(); egCand != EMCands.end(); egCand++) {
    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = (egCand->hwIso() * 4 + egCand->hwQual());

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0, 0, 0, 0);

    int isoFlag = 0;
    bool isinBarrel = false;
    if ((egEtaCut & (1 << eg_eta)) >> eg_eta) {
      isinBarrel = true;
    }

    isoFlag = isinBarrel;
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, isoFlag);
    preSortEGammas.push_back(theEG);
  }

  SortEGammas(&preSortEGammas, &preGtEGammas);
  EGammaToGtScales(params_, &preGtEGammas, egammas);
}
