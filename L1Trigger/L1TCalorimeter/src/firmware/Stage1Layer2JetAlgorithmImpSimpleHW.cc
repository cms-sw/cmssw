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

using namespace std;
using namespace l1t;

Stage1Layer2JetAlgorithmImpSimpleHW::Stage1Layer2JetAlgorithmImpSimpleHW(CaloParamsStage1* params) : params_(params)
{
}

Stage1Layer2JetAlgorithmImpSimpleHW::~Stage1Layer2JetAlgorithmImpSimpleHW(){};

void Stage1Layer2JetAlgorithmImpSimpleHW::processEvent(const std::vector<l1t::CaloRegion> & regions,
						 const std::vector<l1t::CaloEmCand> & EMCands,
						 std::vector<l1t::Jet> * jets){

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  std::vector<l1t::Jet> *preGtJets = new std::vector<l1t::Jet>();

  simpleHWSubtraction(regions, subRegions);
  passThroughJets(subRegions, preGtJets);

  //the jets should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  auto comp = [&](l1t::Jet i, l1t::Jet j)-> bool {
    return (i.hwPt() < j.hwPt() );
  };

  std::sort(preGtJets->begin(), preGtJets->end(), comp);
  std::reverse(preGtJets->begin(), preGtJets->end());


  // drop the 4 LSB before passing to GT
  for(std::vector<l1t::Jet>::const_iterator itJet = preGtJets->begin();
      itJet != preGtJets->end(); ++itJet){
    const unsigned newEta = gtEta(itJet->hwEta());
    const uint16_t rankPt = (itJet->hwPt() >> 4);
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ldummy(0,0,0,0);
    l1t::Jet gtJet(*&ldummy, rankPt, newEta, itJet->hwPhi(), itJet->hwQual());
    jets->push_back(gtJet);
  }

  delete subRegions;
  delete preGtJets;
}
