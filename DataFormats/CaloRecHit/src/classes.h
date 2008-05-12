#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

/* #include <boost/cstdint.hpp>  */

namespace {
    std::vector<reco::CaloCluster> v11;
    std::vector<reco::CaloClusterPtr> v12;
    reco::CaloClusterCollection v1;
    edm::Ptr<reco::CaloCluster> p1;
    edm::PtrVector<reco::CaloCluster> pv1;
    edm::Wrapper<edm::PtrVector<reco::CaloCluster> > wpv1;
}
