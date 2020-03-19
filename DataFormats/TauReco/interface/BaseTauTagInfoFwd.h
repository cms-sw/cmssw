#ifndef DataFormats_TauReco_BaseTauTagInfoFwd_h
#define DataFormats_TauReco_BaseTauTagInfoFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class BaseTauTagInfo;
  /// collection of BaseTauTagInfo objects
  typedef std::vector<BaseTauTagInfo> BaseTauTagInfoCollection;
  /// presistent reference to a BaseTauTagInfo
  typedef edm::Ref<BaseTauTagInfoCollection> BaseTauTagInfoRef;
  /// references to BaseTauTagInfo collection
  typedef edm::RefProd<BaseTauTagInfoCollection> BaseTauTagInfoRefProd;
  /// vector of references to BaseTauTagInfo objects all in the same collection
  typedef edm::RefVector<BaseTauTagInfoCollection> BaseTauTagInfoRefVector;
  /// iterator over a vector of references to BaseTauTagInfo objects all in the same collection
  typedef BaseTauTagInfoRefVector::iterator basetautaginfo_iterator;
}  // namespace reco

#endif
