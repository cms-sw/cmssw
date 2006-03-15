#include "DataFormats/HLTReco/interface/HLTResult.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    reco::HLTResult< 8> h1;
    reco::HLTResult<16> h2;
    reco::HLTResult<24> h3;

    edm::Wrapper<reco::HLTResult< 8> > w1;
    edm::Wrapper<reco::HLTResult<16> > w2;
    edm::Wrapper<reco::HLTResult<24> > w3;
  }
}
