#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Ref.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

namespace {
  struct dictionary {
    std::vector<reco::CaloCluster> v11;
    std::vector<reco::CaloClusterPtr> v12;
    reco::CaloClusterCollection v1;
    std::pair<DetId,float>               hitAndFraction;
    std::vector<std::pair<DetId,float> > hitsAndFractions;
    edm::Ptr<reco::CaloCluster> p1;
    edm::PtrVector<reco::CaloCluster> pv1;
    edm::Wrapper<edm::PtrVector<reco::CaloCluster> > wpv1;
    edm::RefToBase<CaloRecHit> rtb1;
    edm::reftobase::Holder<CaloRecHit, edm::Ref<std::vector<CaloRecHit> > > rb8;
  };
}
