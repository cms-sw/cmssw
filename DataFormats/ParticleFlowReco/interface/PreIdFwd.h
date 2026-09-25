#ifndef PreIdFwd_H
#define PreIdFwd_H
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
namespace reco {
  namespace io_v1 {
    class PreId;
  }
  using PreId = io_v1::PreId;
  typedef std::vector<reco::PreId> PreIdCollection;
  typedef edm::Ref<reco::PreIdCollection> PreIdRef;
}  // namespace reco

#endif
