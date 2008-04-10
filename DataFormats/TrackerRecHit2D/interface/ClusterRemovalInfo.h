#ifndef DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h
#define DataFormats_TrackerRecHit2D_ClusterRemovalInfo_h

#include "DataFormats/Common/interface/DetSetVectorNew.h" 
#include "DataFormats/Provenance/interface/ProductID.h" 

namespace reco {
    
    class ClusterRemovalInfo {
        public:
            typedef std::vector<uint32_t> Indices;

            ClusterRemovalInfo() {}

            ClusterRemovalInfo(const edm::ProductID &pixelProdID, 
                               const edm::ProductID &stripProdID) : 
                stripProdID_(stripProdID), pixelProdID_(pixelProdID) { }
           
            Indices & pixelIndices() { return pixelIndices_; }
            Indices & stripIndices() { return stripIndices_; }

            const Indices & pixelIndices() const { return pixelIndices_; }
            const Indices & stripIndices() const { return stripIndices_; }

            edm::ProductID pixelProdID() const { return pixelProdID_; }
            edm::ProductID stripProdID() const { return stripProdID_; }
            edm::ProductID pixelNewProdID() const { return pixelNewProdID_; }
            edm::ProductID stripNewProdID() const { return stripNewProdID_; }
    
            void setNewPixelProdID(const edm::ProductID &pixelProdID) { pixelNewProdID_ = pixelProdID; }
            void setNewStripProdID(const edm::ProductID &stripProdID) { stripNewProdID_ = stripProdID; }
 
            void swap(reco::ClusterRemovalInfo &other) ;
        private:
            edm::ProductID stripProdID_, pixelProdID_;
            edm::ProductID stripNewProdID_, pixelNewProdID_;
            Indices stripIndices_, pixelIndices_; 
        
     };

    void swap(reco::ClusterRemovalInfo &cri1, reco::ClusterRemovalInfo &cri2) ;

}

#endif
