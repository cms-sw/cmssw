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

unsigned int pack15bits(int pt, int eta, int phi);

Stage1Layer2JetAlgorithmImpSimpleHW::Stage1Layer2JetAlgorithmImpSimpleHW(CaloParamsStage1* params) : params_(params)
{
}

Stage1Layer2JetAlgorithmImpSimpleHW::~Stage1Layer2JetAlgorithmImpSimpleHW(){};

void Stage1Layer2JetAlgorithmImpSimpleHW::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets){

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *preGtJets = new std::vector<l1t::Jet>();
  std::vector<l1t::Jet> *sortedJets = new std::vector<l1t::Jet>();

  //simpleHWSubtraction(regions, subRegions);
  //passThroughJets(subRegions, preGtJets);

  passThroughJets(&regions,preGtJets);

  //the jets should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  // auto comp = [&](l1t::Jet i, l1t::Jet j)-> bool {
  //   return (i.hwPt() < j.hwPt() );
  // };

  // std::sort(preGtJets->begin(), preGtJets->end(), comp);
  // std::reverse(preGtJets->begin(), preGtJets->end());
  // sortedJets = preGtJets;

  SortJets(preGtJets, sortedJets);

  // drop the 4 LSB before passing to GT
  for(std::vector<l1t::Jet>::const_iterator itJet = sortedJets->begin();
      itJet != sortedJets->end(); ++itJet){
    const unsigned newEta = gtEta(itJet->hwEta());
    //const unsigned newEta = itJet->hwEta();
    //std::cout << "pre drop: " << itJet->hwPt();
    const uint16_t rankPt = (itJet->hwPt() >> 4);
    //std::cout << " post drop: " << rankPt << std::endl;
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);
    l1t::Jet gtJet(*&ldummy, rankPt, newEta, itJet->hwPhi(), itJet->hwQual());
    jets->push_back(gtJet);
  }

  int cJets = 0;
  int fJets = 0;
  printf("Central 4x4s\n");
  //printf("pt\teta\tphi\n");
  for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
      itJet != jets->end(); ++itJet){
    if(itJet->hwQual() == 2) continue;
    cJets++;
    unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
    cout << bitset<15>(packed).to_string() << endl;
    if(cJets == 4) break;
  }

  printf("Forward 4x4s\n");
  //printf("pt\teta\tphi\n");
  for(std::vector<l1t::Jet>::const_iterator itJet = jets->begin();
      itJet != jets->end(); ++itJet){
    if(itJet->hwQual() != 2) continue;
    fJets++;
    unsigned int packed = pack15bits(itJet->hwPt(), itJet->hwEta(), itJet->hwPhi());
    cout << bitset<15>(packed).to_string() << endl;
    if(fJets == 4) break;
  }

  delete subRegions;
  delete preGtJets;
  delete sortedJets;
}

unsigned int pack15bits(int pt, int eta, int phi)
{
  return( ((pt & 0x3f)) + ((eta & 0xf) << 6) + ((phi & 0x1f) << 10));
}
