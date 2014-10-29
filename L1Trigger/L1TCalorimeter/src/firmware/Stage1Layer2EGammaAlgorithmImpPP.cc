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


Stage1Layer2EGammaAlgorithmImpPP::Stage1Layer2EGammaAlgorithmImpPP(CaloParamsStage1* params) : params_(params) {};

Stage1Layer2EGammaAlgorithmImpPP::~Stage1Layer2EGammaAlgorithmImpPP(){};



void l1t::Stage1Layer2EGammaAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands, const std::vector<l1t::CaloRegion> & regions, const std::vector<l1t::Jet> * jets, std::vector<l1t::EGamma>* egammas) {


  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::EGamma> *preSortEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *preGtEGammas = new std::vector<l1t::EGamma>();


  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, subRegions, regionPUSParams, regionPUSType);

  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  TwelveByTwelveFinder(0, subRegions, unCorrJets);


  std::cout << "Begin egamma cand debug dump:" << std::endl;
  std::cout << "pt eta phi index assocJetPt lutAddress lutResult" << std::endl;
  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = ((1-egCand->hwIso())*4 + egCand->hwQual()) ;

    //std::cout << "JetRankMax: " << params_->jetScale().rankScaleMax()<< " EmRankMax: " << params_->emScale().rankScaleMax()<< std::endl;
    //std::cout << "JetLinMax: " << params_->jetScale().linScaleMax()<< " EmLinMax: " << params_->emScale().linScaleMax()<< std::endl;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    int isoFlag = 0;
    int ijet_pt=AssociatedJetPt(eg_eta,eg_phi,unCorrJets);
    bool isinBarrel = (eg_eta>=7 && eg_eta<=14);
    unsigned int lutAddress = isoLutIndex(eg_et,ijet_pt);

    if (eg_et >0){
      if (isinBarrel){
	if (lutAddress > params_->egIsolationLUTBarrel()->maxSize()) lutAddress = params_->egIsolationLUTBarrel()->maxSize();
	isoFlag= params_->egIsolationLUTBarrel()->data(lutAddress);
      } else{
	if (lutAddress > params_->egIsolationLUTEndcaps()->maxSize()) lutAddress = params_->egIsolationLUTEndcaps()->maxSize();
	isoFlag= params_->egIsolationLUTEndcaps()->data(lutAddress);
      }
    }

    // std::cout << eg_et << " " << eg_eta << " " << eg_phi << " " << index << " "
    // 	      << ijet_pt << " " << lutAddress << " " << isoFlag << std::endl;
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, isoFlag);
    preSortEGammas->push_back(theEG);
  }

  // printf("Pre-Sort\n");
  // for(std::vector<l1t::EGamma>::const_iterator itEGamma = preSortEGammas->begin();
  //     itEGamma != preSortEGammas->end(); ++itEGamma){
  //   std::cout << itEGamma->hwPt() << " " << itEGamma->hwEta() << " " << itEGamma->hwPhi() << std::endl;
  // }

  SortEGammas(preSortEGammas, preGtEGammas);

  EGammaToGtScales(params_, preGtEGammas, egammas);

  // printf("Post-Sort\n");
  // for(std::vector<l1t::EGamma>::const_iterator itEGamma = egammas->begin();
  //     itEGamma != egammas->end(); ++itEGamma){
  //   std::cout << itEGamma->hwPt() << " " << itEGamma->hwEta() << " " << itEGamma->hwPhi() << std::endl;
  // }

  const bool verbose = true;
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
    for(std::vector<l1t::EGamma>::const_iterator itEGamma = egammas->begin();
	itEGamma != egammas->end(); ++itEGamma){
      if(itEGamma->hwIso() != 0) continue;
      fEGammas++;
      unsigned int packed = pack15bits(itEGamma->hwPt(), itEGamma->hwEta(), itEGamma->hwPhi());
      cout << bitset<15>(packed).to_string() << endl;
      if(fEGammas == 4) break;
    }
  }

  delete subRegions;
  delete unCorrJets;
  delete preSortEGammas;
  delete preGtEGammas;

}





/// -----  Compute isolation sum --------------------
double l1t::Stage1Layer2EGammaAlgorithmImpPP::Isolation(int ieta, int iphi,
							const std::vector<l1t::CaloRegion> & regions)  const {
  double isolation = 0;

  for(CaloRegionBxCollection::const_iterator region = regions.begin();
      region != regions.end(); region++) {

    int regionPhi = region->hwPhi();
    int regionEta = region->hwEta();
    unsigned int deltaPhi = iphi - regionPhi;
    if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1)
      deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

    unsigned int deltaEta = std::abs(ieta - regionEta);

    if ((deltaPhi + deltaEta) > 0 && deltaPhi < 2 && deltaEta < 2)
      isolation += region->hwPt();
  }

  // set output
  return isolation;
}

//ieta =-28, nrTowers 0 is 0, increases to ieta28, nrTowers=kNrTowersInSum
unsigned l1t::Stage1Layer2EGammaAlgorithmImpPP::isoLutIndex(unsigned int egPt,unsigned int jetPt) const
{
  const unsigned int nbitsEG=6;  // number of bits used for EG bins in LUT file (needed for left shift operation)
  //  const unsigned int nbitsJet=9; // not used but here for info  number of bits used for Jet bins in LUT file

  unsigned int address= (jetPt << nbitsEG) + egPt;
  // std::cout << address << "\t## " << egPt << " " << jetPt << std::endl;
  return address;
}



int l1t::Stage1Layer2EGammaAlgorithmImpPP::AssociatedJetPt(int ieta, int iphi,
							      const std::vector<l1t::Jet> * jets)  const {

  bool Debug=false;

  if (Debug) cout << "Number of jets: " << jets->size() << endl;
  int pt = 0;


  for(JetBxCollection::const_iterator itJet = jets->begin();
      itJet != jets->end(); ++itJet){

    int jetEta = itJet->hwEta();
    int jetPhi = itJet->hwPhi();
    if (Debug) cout << "Matching ETA: " << ieta << " " << jetEta << endl;
    if (Debug) cout << "Matching PHI: " << iphi << " " << jetPhi << endl;
    if ((jetEta == ieta) && (jetPhi == iphi)){
      pt = itJet->hwPt();
      break;
    }
  }

  // set output
  return pt;
}



/// -----  Compute H/E --------------------
double l1t::Stage1Layer2EGammaAlgorithmImpPP::HoverE(int et, int ieta, int iphi,
						     const std::vector<l1t::CaloRegion> & regions)  const {
  int hadronicET = 0;

  for(CaloRegionBxCollection::const_iterator region = regions.begin();
      region != regions.end(); region++) {

    int regionET = region->hwPt();
    int regionPhi = region->hwPhi();
    int regionEta = region->hwEta();

    if(iphi == regionPhi && ieta == regionEta) {
      hadronicET = regionET;
      break;
    }
  }

  hadronicET -= et;

  double hoe = 0.0;

  if( hadronicET >0 && et > 0)
    hoe =  (double) hadronicET / (double) et;

  // set output
  return hoe;
}
