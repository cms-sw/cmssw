#ifndef DATAFORMATS_FTLCLUSTER_FTLCLUSTERCOLLECTION_H
#define DATAFORMATS_FTLCLUSTER_FTLCLUSTERCOLLECTION_H

#include "DataFormats/Common/interface/DetSetVector.h" 
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/FTLRecHit/interface/FTLCluster.h"

typedef edmNew::DetSetVector<FTLCluster> FTLClusterCollection;
typedef edm::Ref<FTLClusterCollection, FTLCluster> FTLClusterRef;
typedef edm::DetSetRefVector<FTLCluster> FTLClusterRefs;
typedef edm::RefProd<FTLClusterCollection> FTLClustersRef;

#endif
