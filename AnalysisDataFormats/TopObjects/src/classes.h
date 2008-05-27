#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiMassSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/JetRejObs.h"

#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::pair<int,double>  dummy0;
    std::vector<std::pair<int, double> > dummy1;
    std::vector<std::vector<std::pair<int, double> >  > dummy2;

    typedef edm::Ref<std::vector<JetRejObs> > JetRejObsRef;
    JetRejObs jro;
    std::vector<JetRejObs> v_jro;
    edm::Wrapper<std::vector<JetRejObs> > w_v_jro;
    edm::Ref<std::vector<JetRejObs> > r_jro;

    std::vector<std::pair<double, double> > v_p_dbl_dbl;
    std::pair<unsigned int, std::vector<unsigned int> > p_uint_vint;
    std::vector<std::pair<std::string, double> > v_p_str_dbl;
    std::vector<std::pair<unsigned int, double> > v_p_uint_dbl;
    std::pair<unsigned int, double> p_uint_dbl;
    std::vector<std::pair<std::string, reco::JetTagRef> > v_p_str_jtr;
    std::pair<std::string, reco::JetTagRef> p_str_jtr;
    std::map<TtSemiEvent::HypoKey, reco::NamedCompositeCandidate> m_key_hyp;

    TtGenEvent ttgen;
    StGenEvent stgen;
    TopGenEvent topgen;
    edm::Wrapper<TtGenEvent> w_ttgen;
    edm::Wrapper<StGenEvent> w_stgen;
    edm::Wrapper<TopGenEvent> w_topgen;
    edm::RefProd<TtGenEvent> rp_ttgen;
    edm::RefProd<StGenEvent> rp_stgen;
    edm::RefProd<TopGenEvent> rp_topgen;

    TtSemiEvent ttsemievt;
    TtDilepEvtSolution ttdilep;
    TtSemiEvtSolution ttsemi;
    TtSemiMassSolution ttsemimass;
    TtHadEvtSolution tthad;
    StEvtSolution st;
    std::vector<TtSemiEvent> v_ttsemievt;
    std::vector<TtDilepEvtSolution> v_ttdilep;
    std::vector<TtSemiEvtSolution> v_ttsemi;
    std::vector<TtHadEvtSolution> v_tthad;
    std::vector<StEvtSolution> v_st;
    edm::Wrapper<std::vector<TtSemiEvent> > w_v_ttsemievt;
    edm::Wrapper<std::vector<TtDilepEvtSolution> > w_v_ttdilep;
    edm::Wrapper<std::vector<TtSemiEvtSolution> > w_v_ttsemi;
    edm::Wrapper<std::vector<TtHadEvtSolution> > w_v_tthad;
    edm::Wrapper<std::vector<StEvtSolution> > w_v_st;    
  }
}
