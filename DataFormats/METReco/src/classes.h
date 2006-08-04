#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<reco::MET> dummy1;
    edm::Wrapper<reco::METCollection> dummy2;
    edm::Wrapper< std::vector<reco::MET> > dummy3;
    std::vector<reco::MET> dummy4;

    CaloMETRef r1;
    CaloMETRefProd rp1;
    CaloMETRefVector rv1;
    edm::Wrapper<reco::CaloMET> dummy5;
    edm::Wrapper<reco::CaloMETCollection> dummy7;
    edm::Wrapper< std::vector<reco::CaloMET> > dummy8;
    std::vector<reco::CaloMET> dummy9;
    edm::reftobase::Holder<reco::Candidate,reco::CaloMETRef> rtb1;

    GenMETRef r2;
    GenMETRefProd rp2;
    GenMETRefVector rv2;
    edm::Wrapper<reco::GenMET> dummy10;
    edm::Wrapper<reco::GenMETCollection> dummy11;
    edm::Wrapper< std::vector<reco::GenMET> > dummy12;
    std::vector<reco::GenMET> dummy13;
    edm::reftobase::Holder<reco::Candidate,reco::GenMETRef> rtb2;
  }
}
