#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
    
namespace reco {
    void ClusterRemovalInfo::swap(reco::ClusterRemovalInfo &other) {
        edm::ProductID tmp;
        tmp = stripProdID_; stripProdID_ = other.stripProdID_; other.stripProdID_ = tmp;
        tmp = pixelProdID_; pixelProdID_ = other.pixelProdID_; other.pixelProdID_ = tmp;
        tmp = stripNewProdID_; stripNewProdID_ = other.stripNewProdID_; other.stripNewProdID_ = tmp;
        tmp = pixelNewProdID_; pixelNewProdID_ = other.pixelNewProdID_; other.pixelNewProdID_ = tmp;
        stripIndices_.swap(other.stripIndices_);
        pixelIndices_.swap(other.pixelIndices_);
    }
    void swap(reco::ClusterRemovalInfo &cri1, reco::ClusterRemovalInfo &cri2) {
        cri1.swap(cri2);
    }
}
