#ifndef PreIdFwd_H
#define PreIdFwd_H
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
namespace reco {
  class PreId;
  typedef std::vector<reco::PreId> PreIdCollection;
  typedef edm::Ref<reco::PreIdCollection> PreIdRef;
}  // namespace reco

#endif
