#ifndef TauReco_TauTagInfoFwd_h
#define TauReco_TauTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class TauTagInfo;
  /// collection of TauTagInfo objects
  typedef std::vector<TauTagInfo> TauTagInfoCollection;
  /// presistent reference to a TauTagInfo
  typedef edm::Ref<TauTagInfoCollection> TauTagInfoRef;
  /// references to TauTagInfo collection
  typedef edm::RefProd<TauTagInfoCollection> TauTagInfoRefProd;
  /// vector of references to TauTagInfo objects all in the same collection
  typedef edm::RefVector<TauTagInfoCollection> TauTagInfoRefVector;
  /// iterator over a vector of references to TauTagInfo objects all in the same collection
  typedef TauTagInfoRefVector::iterator tautaginfo_iterator;
}

#endif
