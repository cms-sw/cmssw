#ifndef DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h
#define DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

namespace reco {

  class ClusterRemovalInfo {
  public:
    typedef SiStripRecHit2D::ClusterRef::product_type SiStripClusters;
    typedef SiPixelRecHit::ClusterRef::product_type SiPixelClusters;
    typedef Phase2TrackerRecHit1D::ClusterRef::product_type Phase2TrackerCluster1Ds;
    typedef edm::RefProd<SiStripClusters> SiStripClusterRefProd;
    typedef edm::RefProd<SiPixelClusters> SiPixelClusterRefProd;
    typedef edm::RefProd<Phase2TrackerCluster1Ds> Phase2TrackerCluster1DRefProd;

    typedef std::vector<uint32_t> Indices;

    ClusterRemovalInfo() {}

    ClusterRemovalInfo(const edm::Handle<SiPixelClusters> &pixelClusters,
                       const edm::Handle<SiStripClusters> &stripClusters)
        : pixelProd_(pixelClusters), stripProd_(stripClusters), phase2OTProd_() {}

    ClusterRemovalInfo(const edm::Handle<SiPixelClusters> &pixelClusters,
                       const edm::Handle<Phase2TrackerCluster1Ds> &phase2OTClusters)
        : pixelProd_(pixelClusters), stripProd_(), phase2OTProd_(phase2OTClusters) {}

    ClusterRemovalInfo(const edm::Handle<SiPixelClusters> &pixelClusters)
        : pixelProd_(pixelClusters), stripProd_(), phase2OTProd_() {}

    ClusterRemovalInfo(const edm::Handle<SiStripClusters> &stripClusters)
        : pixelProd_(), stripProd_(stripClusters), phase2OTProd_() {}

    ClusterRemovalInfo(const edm::Handle<Phase2TrackerCluster1Ds> &phase2OTClusters)
        : pixelProd_(), stripProd_(), phase2OTProd_(phase2OTClusters) {}

    void getOldClustersFrom(const ClusterRemovalInfo &other) {
      stripProd_ = other.stripProd_;
      pixelProd_ = other.pixelProd_;
      phase2OTProd_ = other.phase2OTProd_;
    }

    Indices &pixelIndices() { return pixelIndices_; }
    Indices &stripIndices() { return stripIndices_; }
    Indices &phase2OTIndices() { return phase2OTIndices_; }

    const Indices &pixelIndices() const { return pixelIndices_; }
    const Indices &stripIndices() const { return stripIndices_; }
    const Indices &phase2OTIndices() const { return phase2OTIndices_; }

    const SiPixelClusterRefProd &pixelRefProd() const { return pixelProd_; }
    const SiStripClusterRefProd &stripRefProd() const { return stripProd_; }
    const Phase2TrackerCluster1DRefProd &phase2OTRefProd() const { return phase2OTProd_; }
    const SiPixelClusterRefProd &pixelNewRefProd() const { return pixelNewProd_; }
    const SiStripClusterRefProd &stripNewRefProd() const { return stripNewProd_; }
    const Phase2TrackerCluster1DRefProd &phase2OTNewRefProd() const { return phase2OTNewProd_; }

    void setNewPixelClusters(const edm::OrphanHandle<SiPixelClusters> &pixels) {
      pixelNewProd_ = SiPixelClusterRefProd(pixels);
    }
    void setNewStripClusters(const edm::OrphanHandle<SiStripClusters> &strips) {
      stripNewProd_ = SiStripClusterRefProd(strips);
    }
    void setNewPhase2OTClusters(const edm::OrphanHandle<Phase2TrackerCluster1Ds> &phase2OTs) {
      phase2OTNewProd_ = Phase2TrackerCluster1DRefProd(phase2OTs);
    }

    bool hasPixel() const { return pixelProd_.isNonnull(); }
    bool hasStrip() const { return stripProd_.isNonnull(); }
    bool hasPhase2OT() const { return phase2OTProd_.isNonnull(); }

    void swap(reco::ClusterRemovalInfo &other);

  private:
    SiPixelClusterRefProd pixelProd_;
    SiStripClusterRefProd stripProd_;
    Phase2TrackerCluster1DRefProd phase2OTProd_;
    SiPixelClusterRefProd pixelNewProd_;
    SiStripClusterRefProd stripNewProd_;
    Phase2TrackerCluster1DRefProd phase2OTNewProd_;
    Indices stripIndices_, pixelIndices_, phase2OTIndices_;
  };

  void swap(reco::ClusterRemovalInfo &cri1, reco::ClusterRemovalInfo &cri2);

}  // namespace reco

#endif
