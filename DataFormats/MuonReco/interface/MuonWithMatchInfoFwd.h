#ifndef MuonReco_MuonWithMatchInfoFwd_h
#define MuonReco_MuonWithMatchInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class MuonWithMatchInfo;
  /// collection of MuonWithMatchInfo objects
  typedef std::vector<MuonWithMatchInfo> MuonWithMatchInfoCollection;
  /// presistent reference to a MuonWithMatchInfo
  typedef edm::Ref<MuonWithMatchInfoCollection> MuonWithMatchInfoRef;
  /// references to MuonWithMatchInfo collection
  typedef edm::RefProd<MuonWithMatchInfoCollection> MuonWithMatchInfoRefProd;
  /// vector of references to MuonWithMatchInfo objects all in the same collection
  typedef edm::RefVector<MuonWithMatchInfoCollection> MuonWithMatchInfoRefVector;
  /// iterator over a vector of references to MuonWithMatchInfo objects all in the same collection
  typedef MuonWithMatchInfoRefVector::iterator MuonWithMatchInfo_iterator;
}

#endif
