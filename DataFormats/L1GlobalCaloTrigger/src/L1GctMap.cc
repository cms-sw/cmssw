
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctMap.h"

L1GctMap* L1GctMap::m_instance = 0;

/// constructor
L1GctMap::L1GctMap() { 

}

/// destructor
L1GctMap::~L1GctMap() {

}

/// get the RCT crate number
unsigned L1GctMap::rctCrate(L1GctRegion r) {
  return 0;
}

/// get the SC number
unsigned L1GctMap::sourceCard(L1GctRegion r) {
  return 0;
}

/// get the eta index within an RCT crate
unsigned L1GctMap::rctEta(L1GctRegion r) {
  return 0;
}

/// get the phi index within an RCT crate
unsigned L1GctMap::rctPhi(L1GctRegion r) {
  return 0;
}

/// get global eta index
unsigned L1GctMap::eta(L1GctRegion r) {
  return 0;
}

/// get global phi index 
unsigned L1GctMap::phi(L1GctRegion r) {
  return 0;
}

/// get physical eta 
//double L1GctMap::eta(L1GctRegion r) {
//
//}

/// get physical phi
//double L1GctMap::phi(L1GctRegion r) {
//
//}

/// get ID from eta, phi indices
unsigned L1GctMap::id(unsigned ieta, unsigned iphi) {
  return 0;
}
