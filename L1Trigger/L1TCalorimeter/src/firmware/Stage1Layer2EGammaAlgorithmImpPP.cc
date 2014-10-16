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


  double egLsb=params_->egLsb();
  double jetLsb=params_->jetLsb();
  int egSeedThreshold= floor( params_->egSeedThreshold()/egLsb + 0.5);
  int jetSeedThreshold= floor( params_->jetSeedThreshold()/jetLsb + 0.5);
  // double egRelativeJetIsolationBarrelCut = params_->egRelativeJetIsolationBarrelCut();
  // double egRelativeJetIsolationEndcapCut = params_->egRelativeJetIsolationEndcapCut();
  unsigned int egRelativeJetIsolationBarrelCut = floor( params_->egRelativeJetIsolationBarrelCut()*100 +0.5);
  unsigned int egRelativeJetIsolationEndcapCut = floor( params_->egRelativeJetIsolationEndcapCut()*100 +0.5);
  int egMinPtRelativeJetIsolation = params_->egMinPtRelativeJetIsolation();
  int egMaxPtRelativeJetIsolation = params_->egMaxPtRelativeJetIsolation();
  int egMinPt3x3HoE = params_->egMinPt3x3HoE();
  int egMaxPt3x3HoE = params_->egMaxPt3x3HoE();

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
  // slidingWindowJetFinder(jetSeedThreshold, subRegions, unCorrJets);
  TwelveByTwelveFinder(jetSeedThreshold, subRegions, unCorrJets);


  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = egCand->hwQual();

    //std::cout << "JetRankMax: " << params_->jetScale().rankScaleMax()<< " EmRankMax: " << params_->emScale().rankScaleMax()<< std::endl;
    //std::cout << "JetLinMax: " << params_->jetScale().linScaleMax()<< " EmLinMax: " << params_->emScale().linScaleMax()<< std::endl;
    if(eg_et <= egSeedThreshold) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    //int quality = 1;
    int isoFlag = 0;
    int isoFlagRct = 0;

    // 3x3 HoE, computed in 3x3
    if(eg_et>=egMinPt3x3HoE && eg_et < egMaxPt3x3HoE ) {
                 if(egCand->hwIso()) isoFlagRct =1;
    }
    else {isoFlagRct =1;}   

    int ijet_pt=AssociatedJetPt(eg_eta,eg_phi,unCorrJets);
    // double jet_pt=ijet_pt*jetLsb;
    bool isinBarrel = (eg_eta>=7 && eg_eta<=14);
    if (ijet_pt>0 && eg_et>=egMinPtRelativeJetIsolation && eg_et<egMaxPtRelativeJetIsolation){

      // double jetIsolationEG = jet_pt - eg_et;        // Jet isolation
      // double relativeJetIsolationEG = jetIsolationEG / eg_et;

      // if (relativeJetIsolationEG*100<100)
      // 	std::cout << "eg/jet/isol/relisol: " << eg_et << " / " << jet_pt << " /\t " << isol << " / " << int(relativeJetIsolationEG*100+0.5) << "\t address: " << lutAddress << std::endl;
      //
      // if(eg_et >0 && eg_et<63 && isinBarrel && relativeJetIsolationEG < egRelativeJetIsolationBarrelCut) isoFlag=1;
      // if(eg_et >0 && eg_et<63 && !isinBarrel && relativeJetIsolationEG < egRelativeJetIsolationEndcapCut) isoFlag=1;
      // if( eg_et >= 63) isoFlag=1;

      unsigned int lutAddress = isoLutIndex(eg_et,ijet_pt);
      if (lutAddress > params_->egIsolationLUT()->maxSize()) lutAddress = params_->egIsolationLUT()->maxSize();

      unsigned int isol= params_->egIsolationLUT()->data(lutAddress);
      if(eg_et >0 && isinBarrel  && isol < egRelativeJetIsolationBarrelCut) isoFlag=1;
      if(eg_et >0 && !isinBarrel && isol < egRelativeJetIsolationEndcapCut) isoFlag=1;

    }else{ // no associated jet; assume it's an isolated eg
      isoFlag=1;
    }

    int fullIsoFlag=isoFlag*isoFlagRct;

    // double hoe = HoverE(eg_et, eg_eta, eg_phi, *subRegions);
    //if(eg_et>20)
    //std::cout << "eg/jet/isol/relisol: " << eg_et << " / " << ijet_pt << " /\t " << isoFlag <<  "    "<< isoFlagRct << "    "<<fullIsoFlag<<"   "<<egCand->hwIso()<<std::endl;

    // ------- fill the EG candidate vector ---------
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, fullIsoFlag);
    //?? if( hoe < HoverECut) egammas->push_back(theEG);
    preSortEGammas->push_back(theEG);
  }

  SortEGammas(preSortEGammas, preGtEGammas);

  EGammaToGtScales(params_, preGtEGammas, egammas);

  const bool verbose = false;
  if(verbose)
  {
    int cEGammas = 0;
    int fEGammas = 0;
    printf("Isolated\n");
    for(std::vector<l1t::EGamma>::const_iterator itEGamma = egammas->begin();
	itEGamma != egammas->end(); ++itEGamma){
      if(itEGamma->hwIso() != 1) continue;
      cEGammas++;
      unsigned int packed = pack15bits(itEGamma->hwPt(), itEGamma->hwEta(), itEGamma->hwPhi());
      cout << bitset<15>(packed).to_string() << endl;
      if(cEGammas == 4) break;
    }

    printf("Non-isolated\n");
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
  // const unsigned int kNrTowersInSum=72*params_->egIsoMaxEtaAbsForTowerSum()*2;
  // const unsigned int kTowerGranularity=params_->egIsoPUEstTowerGranularity();
  // const unsigned int kMaxAddress = kNrTowersInSum%kTowerGranularity==0 ? (kNrTowersInSum/kTowerGranularity+1)*28*2 :
  //                                                                       (kNrTowersInSum/kTowerGranularity)*28*2;

  // unsigned int nrTowersNormed = nrTowers/kTowerGranularity;


  // if(std::abs(iEta)>28 || iEta==0 || nrTowers>kNrTowersInSum) return kMaxAddress;
  // else return iEtaNormed*(kNrTowersInSum/kTowerGranularity+1)+nrTowersNormed;

  if (jetPt>511) jetPt=511;
  if (egPt>63) egPt=63;

  return 511*(egPt-1)+(jetPt-1);
}



int l1t::Stage1Layer2EGammaAlgorithmImpPP::AssociatedJetPt(int ieta, int iphi,
							      const std::vector<l1t::Jet> * jets)  const {

  bool Debug=false;

  if (Debug) cout << "Number of jets: " << jets->size() << endl;
  int pt = -1;


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
