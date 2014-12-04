#ifndef DeDxHitInfo_H
#define DeDxHitInfo_H
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Common/interface/Association.h"

namespace reco {
 class DeDxHitInfo
  {
   public:
     class DeDxHitInfoContainer{
        public:
           DeDxHitInfoContainer(){ charge_=0.0f; pathlength_=0.0f; detId_=DetId(0); pos_=LocalPoint(0,0);}
           DeDxHitInfoContainer(const float& charge, const float& pathlength, const DetId& detId, const LocalPoint& pos){
              charge_=charge; pathlength_=pathlength; detId_=detId; pos_=pos;
           }
           float charge(){return charge_;}
           float pathlength(){return pathlength_;}
           DetId detId(){return detId_;}
           LocalPoint pos(){return pos_;}
        private:
           float charge_; float pathlength_; DetId detId_; LocalPoint pos_;
     };
     typedef  std::vector<DeDxHitInfo::DeDxHitInfoContainer>    DeDxHitInfoContainerCollection;

   public:
     DeDxHitInfo(){}
     size_t size(){return infos_.size();}
     float charge(size_t i){return infos_[i].charge();}
     float pathlength(size_t i){return infos_[i].pathlength();}
     DetId detId(size_t i){return infos_[i].detId();}
     LocalPoint pos(size_t i){return infos_[i].pos();}
     SiPixelCluster* pixelCluster(size_t i){size_t P=0; bool isPixel=false; for(size_t j=0;j<=i&&j<infos_.size();j++){if(detId(j).subdetId()< SiStripDetId::TIB){P++; isPixel=true;}else{isPixel=false;}} if(isPixel){return &(pixelClusters_[P]);} return NULL;  }
     SiStripCluster* stripCluster(size_t i){size_t S=0; bool isStrip=false; for(size_t j=0;j<=i&&j<infos_.size();j++){if(detId(j).subdetId()>=SiStripDetId::TIB){S++; isStrip=true;}else{isStrip=false;}} if(isStrip){return &(stripClusters_[S]);} return NULL;  }
     const std::vector<SiStripCluster>& stripClusters(){return stripClusters_;}
     const std::vector<SiPixelCluster>& pixelClusters(){return pixelClusters_;}

     void addHit(const float& charge, const float& pathlength, const DetId& detId, const LocalPoint& pos, const SiStripCluster& stripCluster){
        infos_.push_back(DeDxHitInfoContainer(charge, pathlength, detId, pos));   
        stripClusters_.push_back(stripCluster);
     }
     void addHit(const float& charge, const float& pathlength, const DetId& detId, const LocalPoint& pos, const SiPixelCluster& pixelCluster){
        infos_.push_back(DeDxHitInfoContainer(charge, pathlength, detId, pos));
        pixelClusters_.push_back(pixelCluster);
     }

   private:
     std::vector<DeDxHitInfoContainer> infos_;         
     std::vector<SiStripCluster> stripClusters_;
     std::vector<SiPixelCluster> pixelClusters_;
  };

  typedef  std::vector<DeDxHitInfo>    DeDxHitInfoCollection;
  typedef  edm::Ref<DeDxHitInfoCollection> DeDxHitInfoRef;
  typedef  edm::RefProd<DeDxHitInfoCollection> DeDxHitInfoRefProd;
  typedef  edm::RefVector<DeDxHitInfoCollection> DeDxHitInfoRefVector;
  typedef  edm::Association<DeDxHitInfoCollection> DeDxHitInfoAss;
}

#endif
