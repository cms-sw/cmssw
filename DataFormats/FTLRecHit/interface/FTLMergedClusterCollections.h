#ifndef DataFormats_FTLRecHit_FTLMergedClusterCollections_H
#define DataFormats_FTLRecHit_FTLMergedClusterCollections_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/FTLRecHit/interface/FTLMergedCluster.h"

typedef edmNew::DetSetVector<FTLMergedCluster> FTLMergedClusterCollection;
typedef edm::Ref<FTLMergedClusterCollection, FTLMergedCluster> FTLMergedClusterRef;
typedef edm::DetSetRefVector<FTLMergedCluster> FTLMergedClusterRefs;
typedef edm::RefProd<FTLMergedClusterCollection> FTLMergedClustersRef;

#endif
