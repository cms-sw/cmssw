#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
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

    std::multimap<double,double> m2;
    std::vector<reco::MuIsoDeposit> v2;
    edm::Wrapper<std::vector<reco::MuIsoDeposit> > c2;
    edm::Ref<std::vector<reco::MuIsoDeposit> > r2;
    edm::RefProd<std::vector<reco::MuIsoDeposit> > rp2;
    edm::RefVector<std::vector<reco::MuIsoDeposit> > rv2;

    std::vector<reco::MuIsoSimpleDeposit> v3;
    edm::Wrapper<std::vector<reco::MuIsoSimpleDeposit> > c3;
    edm::Ref<std::vector<reco::MuIsoSimpleDeposit> > r3;
    edm::RefProd<std::vector<reco::MuIsoSimpleDeposit> > rp3;
    edm::RefVector<std::vector<reco::MuIsoSimpleDeposit> > rv3;

    std::map<unsigned int, reco::MuIsoSimpleDeposit> m4;
    reco::MuIsoAssociationMap v4;
    edm::Wrapper<reco::MuIsoAssociationMap> w4;
    edm::helpers::Key<edm::RefProd<std::vector<reco::Track> > > h4;

  }
}
