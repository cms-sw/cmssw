#ifndef RECOHGCAL_TICL_TRACKSTERSPCA_H
#define RECOHGCAL_TICL_TRACKSTERSPCA_H

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"
#include <vector>

namespace ticl {
  void assignPCAtoTracksters(std::vector<Trackster> &,
                             const std::vector<reco::CaloCluster> &,
			     const edm::ValueMap<std::pair<float, float>> &,
                             double,
                             bool energyWeight = true);
}
#endif
