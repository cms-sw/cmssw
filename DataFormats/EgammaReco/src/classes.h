#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/EgammaTrigger.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

namespace { 
  namespace {
    std::vector<reco::EcalRecHitData> v01;
    std::vector<reco::BasicCluster> v11;
    reco::BasicClusterCollection v1;
    edm::Wrapper<reco::BasicClusterCollection> w1;
    edm::Ref<reco::BasicClusterCollection> r1;
    edm::RefProd<reco::BasicClusterCollection> rp1;
    edm::RefVector<reco::BasicClusterCollection> rv1;

    std::vector<reco::SuperCluster> sv3;
    reco::SuperClusterCollection v3;
    edm::Wrapper<reco::SuperClusterCollection> w3;
    edm::Ref<reco::SuperClusterCollection> r3;
    edm::RefProd<reco::SuperClusterCollection> rp3;
    edm::RefVector<reco::SuperClusterCollection> rv3;

    std::vector<reco::EcalCluster> tsv3;

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

    reco::ElectronPixelSeedCollection v111;
    edm::Wrapper<reco::ElectronPixelSeedCollection> w111;
    edm::Ref<reco::ElectronPixelSeedCollection> r111;
    edm::RefProd<reco::ElectronPixelSeedCollection> rp111;
    edm::RefVector<reco::ElectronPixelSeedCollection> rv111;
    
    reco::SeedSuperClusterAssociationCollection v112;
    edm::Wrapper<reco::SeedSuperClusterAssociationCollection> w112;
    reco::SeedSuperClusterAssociation ra112;
    reco::SeedSuperClusterAssociationRef r112;
    reco::SeedSuperClusterAssociationRefProd rp112;
    reco::SeedSuperClusterAssociationRefVector rv112;
    
    reco::PreshowerClusterCollection ps5;
    edm::Wrapper<reco::PreshowerClusterCollection> psw5;
    edm::Ref<reco::PreshowerClusterCollection> psr5;
    edm::RefProd<reco::PreshowerClusterCollection> psrp5;
    edm::RefVector<reco::PreshowerClusterCollection> psrv5;

  }
}
