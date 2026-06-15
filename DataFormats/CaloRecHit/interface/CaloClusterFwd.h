#ifndef CaloRecHit_CaloClusterFwd_h
#define CaloRecHit_CaloClusterFwd_h

namespace edm {
  template <typename T>
  class View;
}

namespace reco {
  namespace io_v1 {
    class CaloCluster;
  }
  using CaloCluster = io_v1::CaloCluster;
}  // namespace reco

#endif
