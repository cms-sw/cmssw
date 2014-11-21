#ifndef DeDxHitInfo_H
#define DeDxHitInfo_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include <vector>

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

namespace reco {

 class DeDxHitInfo
  {
   public:
     std::vector<float> charges;
     std::vector<float> pathlengths;
     std::vector<uint32_t> detIds;
     std::vector<float> localPosXs;
     std::vector<float> localPosYs;
     std::vector<SiStripCluster> stripClusters;
     std::vector<SiPixelCluster> pixelClusters;

     DeDxHitInfo(){}
  };

  typedef  std::vector<DeDxHitInfo>    DeDxHitInfoCollection;
  typedef  edm::ValueMap<DeDxHitInfo>  DeDxHitInfoValueMap;  
  typedef  edm::Ref<DeDxHitInfoCollection> DeDxHitInfoRef;
  typedef  edm::RefProd<DeDxHitInfoCollection> DeDxHitInfoRefProd;
  typedef  edm::RefVector<DeDxHitInfoCollection> DeDxHitInfoRefVector;
  typedef  edm::Association<DeDxHitInfoCollection> DeDxHitInfoAss;

}

#endif
