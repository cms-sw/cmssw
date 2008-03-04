#include "DataFormats/HLTReco/interface/HLTResult.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/ModuleTiming.h"
#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    boost::transform_iterator<HLTPerformanceInfo::Path::Adapter,__gnu_cxx::__normal_iterator<const unsigned int*,std::vector<unsigned int> >,boost::use_default,boost::use_default> hltfubar1;
    boost::transform_iterator<HLTPerformanceInfo::Path::Adapter,__gnu_cxx::__normal_iterator<const unsigned long*,std::vector<unsigned long> >,boost::use_default,boost::use_default> hltfubar2;
  }
}

namespace {
  namespace {

    reco::HLTResult< 8> h1;
    reco::HLTResult<16> h2;
    reco::HLTResult<24> h3;

    edm::Wrapper<reco::HLTResult< 8> > w1;
    edm::Wrapper<reco::HLTResult<16> > w2;
    edm::Wrapper<reco::HLTResult<24> > w3;

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

    edm::Wrapper<reco::HLTFilterObjectBase>      whlt1;
    edm::Wrapper<reco::HLTFilterObject>          whlt2;
    edm::Wrapper<reco::HLTFilterObjectWithRefs>  whlt3;
    edm::Wrapper<reco::HLTPathObject>            whlt4;
    edm::Wrapper<reco::HLTGlobalObject>          whlt6;

    edm::EventTime                                et0;

    edm::Wrapper<edm::EventTime>                wet10;

    // Performance Information
    HLTPerformanceInfo pw0;
    edm::Wrapper<HLTPerformanceInfo> pw1;
    HLTPerformanceInfoCollection pw2; 
    edm::Wrapper<HLTPerformanceInfoCollection> pw3; 

    HLTPerformanceInfo::Module pw4;
    HLTPerformanceInfo::Path pw6;
    std::vector<HLTPerformanceInfo::Module> pw8;
    std::vector<HLTPerformanceInfo::Module>::const_iterator pw9;
    std::vector<HLTPerformanceInfo::Path> pw10;
    std::vector<HLTPerformanceInfo::Path>::const_iterator pw11;
    HLTPerformanceInfo::Path::Adapter pw12;
    HLTPerformanceInfo::Path::const_iterator pw13;
  }
}
