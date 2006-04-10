#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/PreShowerCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/EgammaTrigger.h"

namespace { 
  namespace {
    std::vector<reco::EcalRecHitData> v01;
    std::vector<reco::BasicCluster> v11;
    reco::BasicClusterCollection v1;
    edm::Wrapper<reco::BasicClusterCollection> w1;
    edm::Ref<reco::BasicClusterCollection> r1;
    edm::RefProd<reco::BasicClusterCollection> rp1;
    edm::RefVector<reco::BasicClusterCollection> rv1;

    reco::PreShowerClusterCollection v2;
    edm::Wrapper<reco::PreShowerClusterCollection> w2;
    edm::Ref<reco::PreShowerClusterCollection> r2;
    edm::RefProd<reco::PreShowerClusterCollection> rp2;
    edm::RefVector<reco::PreShowerClusterCollection> rv2;

    std::vector<reco::SuperCluster> sv3;
    reco::SuperClusterCollection v3;
    edm::Wrapper<reco::SuperClusterCollection> w3;
    edm::Ref<reco::SuperClusterCollection> r3;
    edm::RefProd<reco::SuperClusterCollection> rp3;
    edm::RefVector<reco::SuperClusterCollection> rv3;

    reco::EgammaTriggerCollection v4;
    edm::Wrapper<reco::EgammaTriggerCollection> w4;
    edm::Ref<reco::EgammaTriggerCollection> r4;
    edm::RefProd<reco::EgammaTriggerCollection> rp4;
    edm::RefVector<reco::EgammaTriggerCollection> rv4;

    reco::ClusterShapeCollection v5;
    edm::Wrapper<reco::ClusterShapeCollection> w5;
    edm::Ref<reco::ClusterShapeCollection> r5;
    edm::RefProd<reco::ClusterShapeCollection> rp5;
    edm::RefVector<reco::ClusterShapeCollection> rv5;

    reco::ClusterPi0DiscriminatorCollection v6;
    edm::Wrapper<reco::ClusterPi0DiscriminatorCollection> w6;
    edm::Ref<reco::ClusterPi0DiscriminatorCollection> r6;
    edm::RefProd<reco::ClusterPi0DiscriminatorCollection> rp6;
    edm::RefVector<reco::ClusterPi0DiscriminatorCollection> rv6;
  }
}
