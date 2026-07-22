#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "DataFormats/FTLRecHit/interface/FTLCluster.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

// FTLMergedClusters
#include "DataFormats/FTLRecHit/interface/FTLMergedClusterCollections.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/FTLRecHit/interface/FTLTrackingRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLSeverityLevel.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Holder.h"
#include <vector>

//raw to rechit specific formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitComparison.h"

namespace DataFormats_FTLRecHit {
  struct dictionary {
    std::vector<FTLClusterRef> v_clusterRefs;
    edmNew::DetSetVector<std::vector<FTLClusterRef>> dsv_v_clusterRefs;
    edm::Wrapper<edmNew::DetSetVector<std::vector<FTLClusterRef>>> w_dsv_v_clusterRefs;
  };
}  // namespace DataFormats_FTLRecHit
