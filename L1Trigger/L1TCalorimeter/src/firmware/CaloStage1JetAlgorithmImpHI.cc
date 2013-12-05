///
/// \class l1t::CaloStage1JetAlgorithmImpHI
///
/// Implementation:
/// Demonstrates how to implement firmware.
///
/// \author: R. Alex Barbieri MIT
///

// This example implemenents firmware version 1 and 2.

#include "CaloStage1JetAlgorithmImp.h"

using namespace std;
using namespace l1t;

CaloStage1JetAlgorithmImpHI::CaloStage1JetAlgorithmImpHI(const CaloParams & dbPars) : db(dbPars) {}

CaloStage1JetAlgorithmImpHI::~CaloStage1JetAlgorithmImpHI(){};

void CaloStage1JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
					       std::vector<l1t::Jet> & jets){
  std::vector<l1t::CaloRegion>::const_iterator incell;
  std::vector<l1t::Jet> outcell;

  for (incell = input.begin(); incell != input.end(); ++incell){

    if (db.firmwareVersion() == 1) {
      // firmware version 1: et(out) = A * et + B
      outcell.setEt( db.paramA() * incell->et() + db.paramB() );
    } else {
      // firmware version 2: et(out) = A * et + B * yvar
      outcell.setEt( db.paramA() * incell->et() + db.paramB() * incell->yvar() );
    }
    // both version yvar(out) = yvar
    outcell.setYvar(incell->yvar());

    out.push_back(outcell);

  }


}
