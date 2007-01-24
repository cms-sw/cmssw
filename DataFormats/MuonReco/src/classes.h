#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonWithMatchInfo.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>
#include <map>

namespace {
  namespace {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
    edm::Ref<std::vector<reco::Muon> > r1;
    edm::RefProd<std::vector<reco::Muon> > rp1;
    edm::RefVector<std::vector<reco::Muon> > rv1;
    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> rb1;

    std::vector<reco::MuIsoDeposit> v2;
    edm::Wrapper<std::vector<reco::MuIsoDeposit> > c2;
    edm::Ref<std::vector<reco::MuIsoDeposit> > r2;
    edm::RefProd<std::vector<reco::MuIsoDeposit> > rp2;
    edm::RefVector<std::vector<reco::MuIsoDeposit> > rv2;

    std::vector<reco::MuonWithMatchInfo> v3;
    edm::Wrapper<std::vector<reco::MuonWithMatchInfo> > c3;
    edm::Ref<std::vector<reco::MuonWithMatchInfo> > r3;
    edm::RefProd<std::vector<reco::MuonWithMatchInfo> > rp3;
    edm::RefVector<std::vector<reco::MuonWithMatchInfo> > rv3;
    std::vector<reco::MuonWithMatchInfo::MuonChamberMatch> vmm1;
    std::vector<reco::MuonWithMatchInfo::MuonSegmentMatch> vmm2;

    reco::MuIsoAssociationMap v4;
    edm::Wrapper<reco::MuIsoAssociationMap> w4;

    std::map<unsigned int, reco::MuIsoDeposit> m5;
    reco::MuIsoDepositAssociationMap v5;
    edm::Wrapper<reco::MuIsoDepositAssociationMap> w5;

  }
}
