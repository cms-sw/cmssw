#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

namespace DataFormats_CaloRecHit {
  struct dictionary {
    // FIXME: The following 2 entries are found already in DataFormats/EgammaReco with the typedef 'reco::BasicCluster'
//     std::vector<reco::CaloCluster> v11;
//     edm::Wrapper<std::vector<reco::CaloCluster> > wv11;
    edm::ValueMap<reco::CaloCluster> vmv11;
    edm::Wrapper<edm::ValueMap<reco::CaloCluster> > wvmv11;
    std::vector<reco::CaloClusterPtr> v12;
//     reco::CaloClusterCollection v1; // ambiguity with std::vector<reco::CaloCluster> v11
    edm::Wrapper<std::vector<std::pair<unsigned long,edm::Ptr<reco::CaloCluster> > > > wveepsassocold;
    std::vector<std::pair<unsigned long,edm::Ptr<reco::CaloCluster> > > veepsassocold;
    std::pair<unsigned long, edm::Ptr<reco::CaloCluster> > eepsassocold;
    std::pair<DetId,float>               hitAndFraction;
    std::vector<std::pair<DetId,float> > hitsAndFractions;
    edm::Ptr<reco::CaloCluster> p1;
    edm::PtrVector<reco::CaloCluster> pv1;
    edm::Wrapper<edm::PtrVector<reco::CaloCluster> > wpv1;
    edm::RefToBase<CaloRecHit> rtb1;
    edm::reftobase::Holder<CaloRecHit, edm::Ref<std::vector<CaloRecHit> > > rb8;
  };
}
