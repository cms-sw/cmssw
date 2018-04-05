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


Stage1Layer2EGammaAlgorithmImpHW::Stage1Layer2EGammaAlgorithmImpHW(CaloParamsHelper* params) : params_(params) {};

Stage1Layer2EGammaAlgorithmImpHW::~Stage1Layer2EGammaAlgorithmImpHW(){};



void l1t::Stage1Layer2EGammaAlgorithmImpHW::processEvent(const std::vector<l1t::CaloEmCand> & EMCands, const std::vector<l1t::CaloRegion> & regions, const std::vector<l1t::Jet> * jets, std::vector<l1t::EGamma>* egammas) {


  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::EGamma> *preSortEGammas = new std::vector<l1t::EGamma>();
  std::vector<l1t::EGamma> *preGtEGammas = new std::vector<l1t::EGamma>();


  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, subRegions, params_);

  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  TwelveByTwelveFinder(0, subRegions, unCorrJets);


  for(CaloEmCandBxCollection::const_iterator egCand = EMCands.begin();
      egCand != EMCands.end(); egCand++) {

    int eg_et = egCand->hwPt();
    int eg_eta = egCand->hwEta();
    int eg_phi = egCand->hwPhi();
    int index = (egCand->hwIso()*4 + egCand->hwQual()) ;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > egLorentz(0,0,0,0);

    int isoFlag = 0;
    int ijet_pt=AssociatedJetPt(eg_eta,eg_phi,unCorrJets);
    bool isinBarrel = (eg_eta>=7 && eg_eta<=14);
    unsigned int lutAddress = isoLutIndex(eg_et,ijet_pt);

    // Combined Barrel/Endcap LUT uses upper bit to indicate Barrel / Endcap:
    enum {MAX_LUT_ADDRESS = 0x7fff};
    enum {LUT_BARREL_OFFSET = 0x0, LUT_ENDCAP_OFFSET = 0x8000};
    enum {LUT_RCT_OFFSET = 0x10000};

    unsigned int rct_offset=0;
    if (egCand->hwIso()) rct_offset=LUT_RCT_OFFSET;

    if (eg_et >0){
      if (lutAddress > MAX_LUT_ADDRESS) lutAddress = MAX_LUT_ADDRESS;

      if (isinBarrel){
        isoFlag= params_->egIsolationLUT()->data(LUT_BARREL_OFFSET + rct_offset + lutAddress);
      } else{
        isoFlag= params_->egIsolationLUT()->data(LUT_ENDCAP_OFFSET + rct_offset + lutAddress);
      }
    }

    l1t::EGamma theEG(*&egLorentz, eg_et, eg_eta, eg_phi, index, isoFlag);
    preSortEGammas->push_back(theEG);
  }

  SortEGammas(preSortEGammas, preGtEGammas);

  EGammaToGtScales(params_, preGtEGammas, egammas);

  delete subRegions;
  delete unCorrJets;
  delete preSortEGammas;
  delete preGtEGammas;

}

//ieta =-28, nrTowers 0 is 0, increases to ieta28, nrTowers=kNrTowersInSum
unsigned l1t::Stage1Layer2EGammaAlgorithmImpHW::isoLutIndex(unsigned int egPt,unsigned int jetPt) const
{
  const unsigned int nbitsEG=6;  // number of bits used for EG bins in LUT file (needed for left shift operation)
  //  const unsigned int nbitsJet=9; // not used but here for info  number of bits used for Jet bins in LUT file

  //jetPt &= 511; // Take only the LSB 9 bits to match firmware.
  if(jetPt > 511) jetPt = 511;
  unsigned int address= (jetPt << nbitsEG) + egPt;
  return address;
}

int l1t::Stage1Layer2EGammaAlgorithmImpHW::AssociatedJetPt(int ieta, int iphi,
							      const std::vector<l1t::Jet> * jets)  const {

  int pt = 0;


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
