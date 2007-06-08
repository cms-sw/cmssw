#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>
#include <map>

namespace {
  namespace {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
    edm::Ref<std::vector<reco::Muon> > r1;
    edm::RefProd<std::vector<reco::Muon> > rp1;
    edm::Wrapper<edm::RefVector<std::vector<reco::Muon> > > wrv1;
    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> rb1;

    std::multimap<muonisolation::Direction::Distance,float> v2b;
    std::vector<reco::MuIsoDeposit> v2;
    edm::Wrapper<std::vector<reco::MuIsoDeposit> > c2;
    edm::Ref<std::vector<reco::MuIsoDeposit> > r2;
    edm::RefProd<std::vector<reco::MuIsoDeposit> > rp2;
    edm::Wrapper<edm::RefVector<std::vector<reco::MuIsoDeposit> > > wrv2;
    
    reco::MuonIsolation rmi;

    std::vector<reco::MuonChamberMatch> vmm1;
    std::vector<reco::MuonSegmentMatch> vmm2;

    reco::MuIsoAssociationMap v4;
    edm::Wrapper<reco::MuIsoAssociationMap> w4;

    std::map<unsigned int, reco::MuIsoDeposit> m5;
    reco::MuIsoDepositAssociationMap v5;
    edm::Wrapper<reco::MuIsoDepositAssociationMap> w5;

    reco::MuIsoIntAssociationMap v6;
    edm::Wrapper<reco::MuIsoIntAssociationMap> w6;

    reco::MuIsoFloatAssociationMap v7;
    edm::Wrapper<reco::MuIsoFloatAssociationMap> w7;
   
    std::vector<reco::MuonTrackLinks> tl1;
    edm::Wrapper<std::vector<reco::MuonTrackLinks> > tl2;
    edm::Ref<std::vector<reco::MuonTrackLinks> > tl3;
    edm::RefProd<std::vector<reco::MuonTrackLinks> > tl4;
    edm::Wrapper<edm::RefVector<std::vector<reco::MuonTrackLinks> > > wtl5;

  }
}
