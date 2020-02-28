#ifndef DataFormats_L1Trigger_L1TObjComparison_h
#define DataFormats_L1Trigger_L1TObjComparison_h

#include <utility>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  template <typename T>
  using ObjectRef = edm::Ref<BXVector<T>>;
  template <typename T>
  using ObjectRefBxCollection = BXVector<ObjectRef<T>>;
  template <typename T>
  using ObjectRefPair = std::pair<edm::Ref<BXVector<T>>, edm::Ref<BXVector<T>>>;
  template <typename T>
  using ObjectRefPairBxCollection = BXVector<ObjectRefPair<T>>;
}  // namespace l1t

#endif
