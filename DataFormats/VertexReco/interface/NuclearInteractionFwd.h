#ifndef NuclearInteractionReco_Fwd_h
#define NuclearInteractionReco_Fwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class NuclearInteraction;
  /// collection of NuclearInteractions
  typedef std::vector<NuclearInteraction> NuclearInteractionCollection;
  /// persistent reference to a NuclearInteraction
  typedef edm::Ref<NuclearInteractionCollection> NuclearInteractionRef;
  /// vector of reference to Track in the same collection
  typedef edm::RefVector<NuclearInteractionCollection> NuclearInteractionRefVector;
  /// iterator over a vector of reference to Track in the same collection
  typedef NuclearInteractionRefVector::iterator NuclearInteraction_iterator;
}  // namespace reco

#endif
