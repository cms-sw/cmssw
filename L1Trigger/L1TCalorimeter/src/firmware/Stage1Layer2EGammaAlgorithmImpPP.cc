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


Stage1Layer2EGammaAlgorithmImpPP::Stage1Layer2EGammaAlgorithmImpPP(CaloParamsHelper* params) : params_(params) {};

Stage1Layer2EGammaAlgorithmImpPP::~Stage1Layer2EGammaAlgorithmImpPP(){};



void l1t::Stage1Layer2EGammaAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands, const std::vector<l1t::CaloRegion> & regions, const std::vector<l1t::Jet> * jets, std::vector<l1t::EGamma>* egammas) {


  double egLsb=params_->egLsb();
  double jetLsb=params_->jetLsb();
  int egSeedThreshold= floor( params_->egSeedThreshold()/egLsb + 0.5);
  int jetSeedThreshold= floor( params_->jetSeedThreshold()/jetLsb + 0.5);
  int egMinPtJetIsolation = params_->egMinPtJetIsolation();
  int egMaxPtJetIsolation = params_->egMaxPtJetIsolation();
  int egMinPtHOverEIsolation = params_->egMinPtHOverEIsolation();
  int egMaxPtHOverEIsolation = params_->egMaxPtHOverEIsolation();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::EGamma> *preSortEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *preGtEGammas = new std::vector<l1t::EGamma>();


  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, subRegions, params_);

  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  // slidingWindowJetFinder(jetSeedThreshold, subRegions, unCorrJets);
  TwelveByTwelveFinder(jetSeedThreshold, subRegions, unCorrJets);


  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = (egCand->hwIso()*4 + egCand->hwQual()) ;

    if(eg_et <= egSeedThreshold) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    //int quality = 1;
    int isoFlag = 0;
    int isoFlagRct = 0;

    // 3x3 HoE, computed in 3x3
    if(eg_et>=egMinPtHOverEIsolation && eg_et < egMaxPtHOverEIsolation ) {
                 if(egCand->hwIso()) isoFlagRct =1;
    }
    else {isoFlagRct =1;}

    int ijet_pt=AssociatedJetPt(eg_eta,eg_phi,unCorrJets);
    // double jet_pt=ijet_pt*jetLsb;
    bool isinBarrel = (eg_eta>=7 && eg_eta<=14);
    if (ijet_pt>0 && eg_et>=egMinPtJetIsolation && eg_et<egMaxPtJetIsolation){

      // Combined Barrel/Endcap LUT uses upper bit to indicate Barrel / Endcap:
      enum {MAX_LUT_ADDRESS = 0x7fff};
      enum {LUT_BARREL_OFFSET = 0x0, LUT_ENDCAP_OFFSET = 0x8000};

      unsigned int lutAddress = isoLutIndex(eg_et,ijet_pt);

      if (eg_et >0){
	if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;
	if (isinBarrel){
	  isoFlag= params_->egIsolationLUT()->data(LUT_BARREL_OFFSET + lutAddress);
	} else{
	  isoFlag= params_->egIsolationLUT()->data(LUT_ENDCAP_OFFSET + lutAddress);
	}
      }


    }else{ // no associated jet; assume it's an isolated eg
      isoFlag=1;
    }

    int fullIsoFlag=isoFlag*isoFlagRct;

    // ------- fill the EG candidate vector ---------
    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, fullIsoFlag);
    //?? if( hoe < HoverECut) egammas->push_back(theEG);
    preSortEGammas->push_back(theEG);
  }

  SortEGammas(preSortEGammas, preGtEGammas);

  EGammaToGtScales(params_, preGtEGammas, egammas);

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
///    unsigned int deltaPhi = iphi - regionPhi;
    int deltaPhi = iphi - regionPhi;
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
  return address;
}



int l1t::Stage1Layer2EGammaAlgorithmImpPP::AssociatedJetPt(int ieta, int iphi,
							      const std::vector<l1t::Jet> * jets)  const {

  int pt = -1;


  for(JetBxCollection::const_iterator itJet = jets->begin();
      itJet != jets->end(); ++itJet){

    int jetEta = itJet->hwEta();
    int jetPhi = itJet->hwPhi();
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
