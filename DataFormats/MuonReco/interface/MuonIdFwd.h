#ifndef MuonReco_MuonIdFwd_h
#define MuonReco_MuonIdFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class MuonId;
  /// collection of MuonId objects
  typedef std::vector<MuonId> MuonIdCollection;
  /// presistent reference to a MuonId
  typedef edm::Ref<MuonIdCollection> MuonIdRef;
  /// references to TAMuon collection
  typedef edm::RefProd<MuonIdCollection> MuonIdRefProd;
  /// vector of references to TAMuon objects all in the same collection
  typedef edm::RefVector<MuonIdCollection> MuonIdRefVector;
  /// iterator over a vector of references to MuonId objects all in the same collection
  typedef MuonIdRefVector::iterator MuonId_iterator;
}

#endif
