

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtScales.h"

L1GctEtScales* L1GctEtScales::m_instance = 0;

L1GctEtScales::L1GctEtScales() {

}


L1GctEtScales::~L1GctEtScales() {

}


double L1GctEtScales::et(L1GctEmCand cand) {

  // this depends on the RCT, right?
  
  return 0.;

}


double L1GctEtScales::et(L1GctCenJet cand) {

  // this depends on the GCT Jet Et LUT contents

  return 0.;

}


double L1GctEtScales::et(L1GctTauJet cand) {

  return 0.;

}


double L1GctEtScales::et(L1GctForJet cand) {

  return 0.;

}

