#ifndef CaloRecHit_CaloClusterCollection_h
#define CaloRecHit_CaloClusterCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

namespace reco {
  typedef std::vector<CaloCluster> CaloClusterCollection;
  typedef edm::Ptr<CaloCluster> CaloClusterPtr;
  typedef edm::PtrVector<CaloCluster> CaloClusterPtrVector;
  typedef edm::View<CaloCluster> CaloClusterView;
  typedef CaloClusterPtrVector::iterator CaloCluster_iterator;
}  // namespace reco

#endif
