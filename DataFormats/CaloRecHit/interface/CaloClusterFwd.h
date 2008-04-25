#ifndef CaloRecHit_CaloClusterCollection_h
#define CaloRecHit_CaloClusterCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"


#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

namespace reco {
  /// collection of CaloCluster objects 
  typedef std::vector<CaloCluster> CaloClusterCollection;
  /// edm references
  typedef edm::Ref<CaloClusterCollection> CaloClusterRef;
  typedef edm::RefVector<CaloClusterCollection> CaloClusterRefVector;
  typedef edm::RefProd<CaloClusterCollection> CaloClusterRefProd;
  typedef edm::Ptr<CaloClusterCollection> CaloClusterPtr;
  typedef edm::PtrVector<CaloClusterCollection> CaloClusterPtrVector;

}
#endif
