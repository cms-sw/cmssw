#ifndef MuonReco_MuonFwd_h
#define MuonReco_MuonFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class Muon;
  /// collection of Muon objects
  typedef std::vector<Muon> MuonCollection;
  /// presistent reference to a Muon
  typedef edm::Ref<MuonCollection> MuonRef;
  /// vector of references to Muon objects all in the same collection
  typedef edm::RefVector<MuonCollection> MuonRefs;
  /// iterator over a vector of references to Muon objects all in the same collection
  typedef MuonRefs::iterator muon_iterator;
}

#endif
