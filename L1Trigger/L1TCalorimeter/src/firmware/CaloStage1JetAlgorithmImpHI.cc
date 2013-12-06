///
/// \class l1t::CaloStage1JetAlgorithmImpHI
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implemenents algorithm version 1 and 2.

#include "CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1JetAlgorithmImpHI::CaloStage1JetAlgorithmImpHI(/*const CaloParams & dbPars*/)/* : db(dbPars)*/ {}

CaloStage1JetAlgorithmImpHI::~CaloStage1JetAlgorithmImpHI(){};

void CaloStage1JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
					       std::vector<l1t::Jet> & jets){
  std::vector<l1t::CaloRegion>::const_iterator incell;
  std::vector<l1t::Jet> outcell;

  for (incell = regions.begin(); incell != regions.end(); ++incell){
    //do nothing for now
  }

}
