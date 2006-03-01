#ifndef MuonReco_MuonExtraFwd_h
#define MuonReco_MuonExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class MuonExtra;
  /// collection of MuonExtra objects
  typedef std::vector<MuonExtra> MuonExtraCollection;
  /// presistent reference to a MuonExtra
  typedef edm::Ref<MuonExtraCollection> MuonExtraRef;
  /// vector of references to MuonExtra objects all in the same collection
  typedef edm::RefVector<MuonExtraCollection> MuonExtraRefs;
  /// iterator over a vector of references to MuonExtra objects all in the same collection
  typedef MuonExtraRefs::iterator muonExtra_iterator;
}

#endif
