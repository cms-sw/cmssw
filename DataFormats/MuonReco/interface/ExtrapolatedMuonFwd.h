#ifndef MuonReco_ExtrapolatedMuonFwd_h
#define MuonReco_ExtrapolatedMuonIdFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ExtrapolatedMuon;
  /// collection of ExtrapolatedMuon objects
  typedef std::vector<ExtrapolatedMuon> ExtrapolatedMuonCollection;
  /// presistent reference to a ExtrapolatedMuon
  typedef edm::Ref<ExtrapolatedMuonCollection> ExtrapolatedMuonRef;
  /// references to ExtrapolatedMuon collection
  typedef edm::RefProd<ExtrapolatedMuonCollection> ExtrapolatedMuonRefProd;
  /// vector of references to ExtrapolatedMuon objects all in the same collection
  typedef edm::RefVector<ExtrapolatedMuonCollection> ExtrapolatedMuonRefVector;
  /// iterator over a vector of references to ExtrapolatedMuon objects all in the same collection
  typedef ExtrapolatedMuonRefVector::iterator ExtrapolatedMuon_iterator;
}

#endif
