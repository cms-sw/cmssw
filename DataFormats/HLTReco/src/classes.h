#include "DataFormats/HLTReco/interface/HLTResult.h"
#include "DataFormats/HLTReco/interface/HLTParticle.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {

    reco::HLTResult< 8> h1;
    reco::HLTResult<16> h2;
    reco::HLTResult<24> h3;

    edm::Wrapper<reco::HLTResult< 8> > w1;
    edm::Wrapper<reco::HLTResult<16> > w2;
    edm::Wrapper<reco::HLTResult<24> > w3;

    reco::HLTParticle                                                          hlt0;
    reco::HLTFilterObjectBase                                                  hlt1;
    reco::HLTFilterObject                                                      hlt2;
    reco::HLTFilterObjectWithRefs                                              hlt3;
    reco::HLTPathObject<reco::HLTFilterObject>                                 hlt4;
    reco::HLTPathObject<reco::HLTFilterObjectWithRefs>                         hlt5;
    reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObject> >         hlt6;
    reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> > hlt7;

    edm::Wrapper<reco::HLTParticle>                                                           whlt1;
    edm::Wrapper<reco::HLTFilterObject>                                                       whlt2;
    edm::Wrapper<reco::HLTFilterObjectWithRefs>                                               whlt3;
    edm::Wrapper<reco::HLTPathObject<reco::HLTFilterObject> >                                 whlt4;
    edm::Wrapper<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> >                         whlt5;
    edm::Wrapper<reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObject> > >         whlt6;
    edm::Wrapper<reco::HLTGlobalObject<reco::HLTPathObject<reco::HLTFilterObjectWithRefs> > > whlt7;

  }
}
