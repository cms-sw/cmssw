#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
    
namespace reco {
    void ClusterRemovalInfo::swap(reco::ClusterRemovalInfo &other) {
        pixelProd_.swap(other.pixelProd_);
        stripProd_.swap(other.stripProd_);
        pixelNewProd_.swap(other.pixelNewProd_);
        stripNewProd_.swap(other.stripNewProd_);
        stripIndices_.swap(other.stripIndices_);
        pixelIndices_.swap(other.pixelIndices_);
    }
    void swap(reco::ClusterRemovalInfo &cri1, reco::ClusterRemovalInfo &cri2) {
        cri1.swap(cri2);
    }
}
