#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"

namespace { 
  namespace {
    std::vector<reco::CastorTower> v11;
    reco::CastorTowerCollection v1;
    edm::Wrapper<reco::CastorTowerCollection> w1;
    edm::Ref<reco::CastorTowerCollection> r1;
    edm::RefProd<reco::CastorTowerCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::CastorTowerCollection> > wrv1;

    std::vector<reco::CastorCell> v22;
    reco::CastorCellCollection v2;
    edm::Wrapper<reco::CastorCellCollection> w2;
    edm::Ref<reco::CastorCellCollection> r2;
    edm::RefProd<reco::CastorCellCollection> rp2;
    edm::Wrapper<edm::RefVector<reco::CastorCellCollection> > wrv2;
    
    std::vector<reco::CastorJet> v33;
    reco::CastorJetCollection v3;
    edm::Wrapper<reco::CastorJetCollection> w3;
    edm::Ref<reco::CastorJetCollection> r3;
    edm::RefProd<reco::CastorJetCollection> rp3;
    edm::Wrapper<edm::RefVector<reco::CastorJetCollection> > wrv3;
    
    std::vector<reco::CastorEgamma> v44;
    reco::CastorEgammaCollection v4;
    edm::Wrapper<reco::CastorEgammaCollection> w4;
    edm::Ref<reco::CastorEgammaCollection> r4;
    edm::RefProd<reco::CastorEgammaCollection> rp4;
    edm::Wrapper<edm::RefVector<reco::CastorEgammaCollection> > wrv4;
  }
}
