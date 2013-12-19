///
/// \class l1t::CaloStage1JetAlgorithmImpHI
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implements algorithm version 1 and 2.

#include "CaloStage1JetAlgorithmImp.h"

// Taken from UCT code. Might not be appropriate. Refers to legacy L1 objects.
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//#include "DataFormats/Candidate/interface/LeafCandidate.h"

using namespace std;
using namespace l1t;

CaloStage1JetAlgorithmImpPP::CaloStage1JetAlgorithmImpPP(/*const CaloParams & dbPars*/)/* : db(dbPars)*/{}
//: regionLSB_(0.5) {}

CaloStage1JetAlgorithmImpPP::~CaloStage1JetAlgorithmImpPP(){};

void puSubtractionPP(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions);


void CaloStage1JetAlgorithmImpPP::processEvent(const std::vector<l1t::CaloRegion> & regions,
					       std::vector<l1t::Jet> & jets){

  std::vector<l1t::CaloRegion> subRegions;
  puSubtractionPP(regions, subRegions);
  findJets(subRegions, jets);

  // std::vector<l1t::CaloRegion>::const_iterator incell;
  // for (incell = regions.begin(); incell != regions.end(); ++incell){
  //   //do nothing for now
  // }

}

// NB PU is not in the physical scale!!  Needs to be multiplied by regionLSB
void puSubtractionPP(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions)
{
  // do nothing for now
}
