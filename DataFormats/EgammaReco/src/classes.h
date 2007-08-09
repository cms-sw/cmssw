#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Math/Cartesian3D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0DiscriminatorFwd.h" 
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Rtypes.h" 
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 
#include "DataFormats/EgammaReco/interface/EcalCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/Math/interface/Point3D.h" 
#include "DataFormats/EgammaReco/interface/EgammaTrigger.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h" 
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h" 
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"

namespace { 
  namespace {
    std::vector<reco::BasicCluster> v11;
    reco::BasicClusterCollection v1;
    edm::Wrapper<reco::BasicClusterCollection> w1;
    edm::Ref<reco::BasicClusterCollection> r1;
    edm::RefProd<reco::BasicClusterCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::BasicClusterCollection> > wrv1;

    std::vector<reco::SuperCluster> sv3;
    reco::SuperClusterCollection v3;
    edm::Wrapper<reco::SuperClusterCollection> w3;
    edm::Ref<reco::SuperClusterCollection> r3;
    edm::RefProd<reco::SuperClusterCollection> rp3;
    edm::Wrapper<edm::RefVector<reco::SuperClusterCollection> > wrv3;

    std::vector<reco::EcalCluster> tsv3;

    reco::EgammaTriggerCollection v4;
    edm::Wrapper<reco::EgammaTriggerCollection> w4;
    edm::Ref<reco::EgammaTriggerCollection> r4;
    edm::RefProd<reco::EgammaTriggerCollection> rp4;
    edm::Wrapper<edm::RefVector<reco::EgammaTriggerCollection> > rv4;

    reco::ClusterShapeCollection v5;
    edm::Wrapper<reco::ClusterShapeCollection> w5;
    edm::Ref<reco::ClusterShapeCollection> r5;
    edm::RefProd<reco::ClusterShapeCollection> rp5;
    edm::Wrapper<edm::RefVector<reco::ClusterShapeCollection> > wrv5;

    reco::HFEMClusterShapeCollection v8;
    edm::Wrapper<reco::HFEMClusterShapeCollection> w8;
    edm::Ref<reco::HFEMClusterShapeCollection> r8;
    edm::RefProd<reco::HFEMClusterShapeCollection> rp8;
    edm::Wrapper<edm::RefVector<reco::HFEMClusterShapeCollection> > wrv8;

    reco::ClusterPi0DiscriminatorCollection v6;
    edm::Wrapper<reco::ClusterPi0DiscriminatorCollection> w6;
    edm::Ref<reco::ClusterPi0DiscriminatorCollection> r6;
    edm::RefProd<reco::ClusterPi0DiscriminatorCollection> rp6;
    edm::Wrapper<edm::RefVector<reco::ClusterPi0DiscriminatorCollection> > wrv6;

    reco::ElectronPixelSeedCollection v111;
    edm::Wrapper<reco::ElectronPixelSeedCollection> w111;
    edm::Ref<reco::ElectronPixelSeedCollection> r111;
    edm::RefProd<reco::ElectronPixelSeedCollection> rp111;
    edm::Wrapper<edm::RefVector<reco::ElectronPixelSeedCollection> > wrv111;
    
    reco::SeedSuperClusterAssociationCollection v112;
    edm::Wrapper<reco::SeedSuperClusterAssociationCollection> w112;
    reco::SeedSuperClusterAssociation ra112;
    reco::SeedSuperClusterAssociationRef r112;
    reco::SeedSuperClusterAssociationRefProd rp112;
    edm::Wrapper<reco::SeedSuperClusterAssociationRefVector> wrv112;
    
    reco::PreshowerClusterCollection ps5;
    edm::Wrapper<reco::PreshowerClusterCollection> psw5;
    edm::Ref<reco::PreshowerClusterCollection> psr5;
    edm::RefProd<reco::PreshowerClusterCollection> psrp5;
    edm::Wrapper<edm::RefVector<reco::PreshowerClusterCollection> > wpsrv5;

    reco::BasicClusterShapeAssociationCollection v7;
    edm::Wrapper<reco::BasicClusterShapeAssociationCollection> w7;
    reco::BasicClusterShapeAssociation va7;
    reco::BasicClusterShapeAssociationRef vr7;
    reco::BasicClusterShapeAssociationRefProd vrp7;
    edm::Wrapper<reco::BasicClusterShapeAssociationRefVector> wvrv7;

    reco::HFEMClusterShapeAssociationCollection v9;
    edm::Wrapper<reco::HFEMClusterShapeAssociationCollection> w9;
    reco::HFEMClusterShapeAssociation va9;
    reco::HFEMClusterShapeAssociationRef vr9;
    reco::HFEMClusterShapeAssociationRefProd vrp9;
    edm::Wrapper<reco::HFEMClusterShapeAssociationRefVector> wvrv9;

  }
}
