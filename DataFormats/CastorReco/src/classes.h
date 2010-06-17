#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

namespace { 
  struct dictionary {
    std::vector<reco::CastorTower> v11;
    reco::CastorTowerCollection v1;
    edm::Wrapper<reco::CastorTowerCollection> w1;
    edm::Ref<reco::CastorTowerCollection> r1;
    edm::RefProd<reco::CastorTowerCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::CastorTowerCollection> > wrv1;
    edm::RefVectorIterator<reco::CastorTowerCollection> rvit1; 
  
    std::vector<reco::CastorCell> v22;
    reco::CastorCellCollection v2;
    edm::Wrapper<reco::CastorCellCollection> w2;
    edm::Ref<reco::CastorCellCollection> r2;
    edm::RefProd<reco::CastorCellCollection> rp2;
    edm::Wrapper<edm::RefVector<reco::CastorCellCollection> > wrv2;
    edm::RefVectorIterator<reco::CastorCellCollection> rvit2;
    //edm::auto_ptr<edm::Ref<std::vector<reco::CastorCell> > > ptrv2;
  
    std::vector<reco::CastorCluster> v33;
    reco::CastorClusterCollection v3;
    edm::Wrapper<reco::CastorClusterCollection> w3;
    edm::Ref<reco::CastorClusterCollection> r3;
    edm::RefProd<reco::CastorClusterCollection> rp3;
    edm::Wrapper<edm::RefVector<reco::CastorClusterCollection> > wrv3;
    edm::RefVectorIterator<reco::CastorClusterCollection> rvit3;
  };
}
