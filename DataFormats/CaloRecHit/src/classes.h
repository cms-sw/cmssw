#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

/* #include <boost/cstdint.hpp>  */

    std::vector<reco::CaloCluster> v11;
    reco::CaloClusterCollection v1;
    edm::Wrapper<reco::CaloClusterCollection> w1;
    edm::Ref<reco::CaloClusterCollection> r1;
    edm::RefProd<reco::CaloClusterCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::CaloClusterCollection> > wrv1;
    edm::Ptr<reco::CaloClusterCollection> p1;
    edm::PtrVector<reco::CaloClusterCollection> pv1;
