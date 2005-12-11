#ifndef MuonReco_MuonFwd_h
#define MuonReco_MuonFwd_h
#include <vector>
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"

namespace reco {
  class Muon;
  typedef std::vector<Muon> MuonCollection;
  typedef edm::Ref<MuonCollection> MuonRef;
  typedef edm::RefVector<MuonCollection> MuonRefs;
  typedef MuonRefs::iterator muon_iterator;
}

#endif
