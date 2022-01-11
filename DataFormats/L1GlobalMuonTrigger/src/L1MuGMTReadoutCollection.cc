// -*- C++ -*-
//
// Package:     DataFormats/L1GlobalMuonTrigger
// Class  :     L1MuGMTReadoutCollection
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 04 Nov 2013 17:08:29 GMT
//

// system include files
#include "oneapi/tbb/concurrent_unordered_map.h"

// user include files
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

static tbb::concurrent_unordered_map<int, L1MuGMTReadoutRecord> s_empty_record_cache;

L1MuGMTReadoutRecord const& L1MuGMTReadoutCollection::getDefaultFor(int bx) {
  // if bx not found return empty readout record
  auto itFound = s_empty_record_cache.find(bx);
  if (itFound == s_empty_record_cache.end()) {
    auto foundPair = s_empty_record_cache.insert(std::make_pair(bx, L1MuGMTReadoutRecord(bx)));
    itFound = foundPair.first;
  }
  return itFound->second;
}
