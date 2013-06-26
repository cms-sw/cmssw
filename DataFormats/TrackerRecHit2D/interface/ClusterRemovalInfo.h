#ifndef DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h
#define DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h

#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Handle.h" 
#include "DataFormats/Common/interface/OrphanHandle.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h" 
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h" 

namespace reco {
    
    class ClusterRemovalInfo {
        public:
            typedef SiStripRecHit2D::ClusterRef::product_type SiStripClusters;
            typedef SiPixelRecHit::ClusterRef::product_type   SiPixelClusters;
            typedef edm::RefProd<SiStripClusters> SiStripClusterRefProd;
            typedef edm::RefProd<SiPixelClusters> SiPixelClusterRefProd;

            typedef std::vector<uint32_t> Indices;

            ClusterRemovalInfo() {}

            ClusterRemovalInfo(const edm::Handle<SiPixelClusters> &pixelClusters, 
                               const edm::Handle<SiStripClusters> &stripClusters) : 
                pixelProd_(pixelClusters), stripProd_(stripClusters) { }

            ClusterRemovalInfo(const edm::Handle<SiPixelClusters> &pixelClusters) : 
                pixelProd_(pixelClusters), stripProd_() { }

            ClusterRemovalInfo(const edm::Handle<SiStripClusters> &stripClusters) : 
                pixelProd_(), stripProd_(stripClusters) { }


            void getOldClustersFrom(const ClusterRemovalInfo &other) {
                stripProd_ = other.stripProd_;
                pixelProd_ = other.pixelProd_;
            }
           
            Indices & pixelIndices() { return pixelIndices_; }
            Indices & stripIndices() { return stripIndices_; }

            const Indices & pixelIndices() const { return pixelIndices_; }
            const Indices & stripIndices() const { return stripIndices_; }

            const SiPixelClusterRefProd & pixelRefProd() const { return pixelProd_; }
            const SiStripClusterRefProd & stripRefProd() const { return stripProd_; }
            const SiPixelClusterRefProd & pixelNewRefProd() const { return pixelNewProd_; }
            const SiStripClusterRefProd & stripNewRefProd() const { return stripNewProd_; }
    
            void setNewPixelClusters(const edm::OrphanHandle<SiPixelClusters> &pixels) { pixelNewProd_ = SiPixelClusterRefProd(pixels); }
            void setNewStripClusters(const edm::OrphanHandle<SiStripClusters> &strips) { stripNewProd_ = SiStripClusterRefProd(strips); }

            bool hasPixel() const { return pixelProd_.isNonnull(); } 
            bool hasStrip() const { return stripProd_.isNonnull(); } 

            void swap(reco::ClusterRemovalInfo &other) ;
        private:
            SiPixelClusterRefProd pixelProd_;
            SiStripClusterRefProd stripProd_;
            SiPixelClusterRefProd pixelNewProd_;
            SiStripClusterRefProd stripNewProd_;
            Indices stripIndices_, pixelIndices_; 
     };

    void swap(reco::ClusterRemovalInfo &cri1, reco::ClusterRemovalInfo &cri2) ;

}

#endif
