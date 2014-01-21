///
/// \class l1t::CaloStage1JetAlgorithmImpHI
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implements algorithm version 1 and 2.

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1JetAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"

// Taken from UCT code. Might not be appropriate. Refers to legacy L1 objects.
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//#include "DataFormats/Candidate/interface/LeafCandidate.h"
//#include <stdio.h>

using namespace std;
using namespace l1t;

CaloStage1JetAlgorithmImpHI::CaloStage1JetAlgorithmImpHI(/*const CaloParams & dbPars*/)/* : db(dbPars)*/ {}
//: regionLSB_(0.5) {}

CaloStage1JetAlgorithmImpHI::~CaloStage1JetAlgorithmImpHI(){};

void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> *subRegions);

void CaloStage1JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
					       std::vector<l1t::Jet> * jets){

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();
  puSubtraction(regions, subRegions);
  slidingWindowJetFinder(subRegions, jets);

  delete subRegions;
}

// NB PU is not in the physical scale!!  Needs to be multiplied by regionLSB
void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> *subRegions)
{
  int puLevelHI[L1CaloRegionDetId::N_ETA];
  double r_puLevelHI[L1CaloRegionDetId::N_ETA];
  int etaCount[L1CaloRegionDetId::N_ETA];
  for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
  {
    puLevelHI[i] = 0;
    r_puLevelHI[i] = 0.0;
    etaCount[i] = 0;
  }

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++){
    r_puLevelHI[region->hwEta()] += region->hwPt();
    etaCount[region->hwEta()]++;
  }

  for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
  {
    puLevelHI[i] = floor(r_puLevelHI[i]/etaCount[i] + 0.5);
  }

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
    int subPt = std::max(0, region->hwPt() - puLevelHI[region->hwEta()]);
    int subEta = region->hwEta();
    int subPhi = region->hwPhi();

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *lorentz =
      new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    CaloRegion newSubRegion(*lorentz, 0, 0, subPt, subEta, subPhi, 0, 0, 0);
    subRegions->push_back(newSubRegion);
  }
}
