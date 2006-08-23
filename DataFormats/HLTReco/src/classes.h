#include "DataFormats/HLTReco/interface/HLTResult.h"
#include "DataFormats/HLTReco/interface/HLTParticle.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/ModuleTiming.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {

    reco::HLTResult< 8> h1;
    reco::HLTResult<16> h2;
    reco::HLTResult<24> h3;

    edm::Wrapper<reco::HLTResult< 8> > w1;
    edm::Wrapper<reco::HLTResult<16> > w2;
    edm::Wrapper<reco::HLTResult<24> > w3;

    reco::HLTParticle                             hlt0;
    reco::HLTFilterObjectBase                     hlt1;
    reco::HLTFilterObject                         hlt2;
    reco::HLTFilterObjectWithRefs                 hlt3;
    reco::HLTPathObject                           hlt4;
    reco::HLTGlobalObject                         hlt6;

    edm::RefProd<reco::HLTFilterObjectBase>       r0;
    edm::RefProd<reco::HLTFilterObject>           r1;
    edm::RefProd<reco::HLTFilterObjectWithRefs>   r2;

    edm::reftobase::Holder<reco::HLTFilterObjectBase, edm::RefProd<reco::HLTFilterObjectBase> > rb0;
    edm::reftobase::Holder<reco::HLTFilterObjectBase, edm::RefProd<reco::HLTFilterObject    > > rb1;
    edm::reftobase::Holder<reco::HLTFilterObjectBase, edm::RefProd<reco::HLTFilterObjectWithRefs> > rb2;

    edm::Wrapper<reco::HLTParticle>              whlt0;
    edm::Wrapper<reco::HLTFilterObjectBase>      whlt1;
    edm::Wrapper<reco::HLTFilterObject>          whlt2;
    edm::Wrapper<reco::HLTFilterObjectWithRefs>  whlt3;
    edm::Wrapper<reco::HLTPathObject>            whlt4;
    edm::Wrapper<reco::HLTGlobalObject>          whlt6;

    edm::EventTime                                et0;

    edm::Wrapper<edm::EventTime>                wet10;
  }
}
