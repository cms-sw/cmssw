#ifndef MuonReco_MuonTimeExtraFwd_h
#define MuonReco_MuonTimeExtraFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {
  class MuonTimeExtra;
  /// collection of MuonTimeExtra objects

  /*  typedef std::vector<MuonTimeExtra> MuonTimeExtraCollection;
  /// presistent reference to a MuonTimeExtra object
  typedef edm::Ref<MuonTimeExtraCollection> MuonTimeExtraRef;
  /// references to a MuonTimeExtra collection
  typedef edm::RefProd<MuonTimeExtraCollection> MuonTimeExtraRefProd;
  /// vector of references to MuonTimeExtra objects all in the same collection
  typedef edm::RefVector<MuonTimeExtraCollection> MuonTimeExtraRefVector;
  /// iterator over a vector of references to MuonTimeExtra objects all in the same collection
  typedef MuonTimeExtraRefVector::iterator muontimeextra_iterator; */

  typedef edm::ValueMap<reco::MuonTimeExtra> MuonTimeExtraMap;

}  // namespace reco

#endif
