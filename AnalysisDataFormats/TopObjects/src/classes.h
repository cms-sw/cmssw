#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/CATopJetTagInfo.h"
#include "DataFormats/JetReco/interface/BasicJet.h"

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtHadEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "TString.h"

namespace {
  namespace {
    std::pair<int,double>  dummy0;
    std::vector<std::pair<int, double> > dummy1;
    std::vector<std::vector<std::pair<int, double> >  > dummy2;

    std::pair<unsigned int, double> p_uint_dbl;
    std::vector<std::pair<double, double> > v_p_dbl_dbl;
    std::vector<std::pair<unsigned int, double> > v_p_uint_dbl;
    std::pair<unsigned int, std::vector<unsigned int> > p_uint_vint;
    std::vector<int> v_int;
    std::vector<std::pair<std::string, double> > v_p_str_dbl;

    TString tstr;
    edm::Wrapper<TString> w_tstr;
    edm::Wrapper<std::vector<std::vector<int> > > w_v_vint;

    typedef edm::Ptr<pat::MET> PtrMet;
    typedef edm::Ptr<pat::Jet> PtrJet;
    typedef edm::Ptr<pat::Muon> PtrMuon;
    typedef edm::Ptr<pat::Electron> PtrElec;
    PtrMet  p_met;
    PtrJet  p_jet;
    PtrMuon p_muon;
    PtrElec p_elec;

    TtGenEvent ttgen;
    StGenEvent stgen;
    TopGenEvent topgen;
    TtEvent ttevt;
    TtFullLeptonicEvent ttfullevt;
    TtSemiLeptonicEvent ttsemievt;
    edm::Wrapper<TtGenEvent> w_ttgen;
    edm::Wrapper<StGenEvent> w_stgen;
    edm::Wrapper<TopGenEvent> w_topgen;
    edm::Wrapper<TtFullLeptonicEvent> w_ttfullevt;
    edm::Wrapper<TtSemiLeptonicEvent> w_ttsemievt;
    edm::Wrapper<reco::CompositeCandidate> ttcompcand;

    edm::RefProd<TtGenEvent> rp_ttgen;
    edm::RefProd<StGenEvent> rp_stgen;
    edm::RefProd<TopGenEvent> rp_topgen;
    edm::RefProd<TtFullLeptonicEvent> rp_ttfullevt;
    edm::RefProd<TtSemiLeptonicEvent> rp_ttsemievt;

    std::pair<WDecay::LeptonType, WDecay::LeptonType> p_lepdecay_lepdecay;
    std::pair<reco::CompositeCandidate, std::vector<int> > p_compcand_vint;
    std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > v_p_compcand_vint;
    edm::Wrapper<std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > > w_v_p_compcand_vint;

    std::map<TtEvent::HypoClassKey, std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > > m_key_v_p_compcand_vint;

    TtDilepEvtSolution ttdilep;
    TtSemiEvtSolution ttsemi;
    TtHadEvtSolution tthad;
    StEvtSolution st;
    std::vector<TtDilepEvtSolution> v_ttdilep;
    std::vector<TtSemiEvtSolution> v_ttsemi;
    std::vector<TtHadEvtSolution> v_tthad;
    std::vector<StEvtSolution> v_st;
    edm::Wrapper<std::vector<TtDilepEvtSolution> > w_v_ttdilep;
    edm::Wrapper<std::vector<TtSemiEvtSolution> > w_v_ttsemi;
    edm::Wrapper<std::vector<TtHadEvtSolution> > w_v_tthad;
    edm::Wrapper<std::vector<StEvtSolution> > w_v_st;   


    edm::reftobase::Holder<reco::Jet, reco::BasicJetRef> hbj;
    edm::reftobase::RefHolder<reco::BasicJetRef> rhbj;

    reco::CATopJetProperties                                            catopjetp;
    std::pair<edm::RefToBase<reco::Jet>, reco::CATopJetProperties>      catopjetp_p;

    reco::CATopJetTagInfo                                               catopjet;
    reco::CATopJetTagInfoCollection                                     catopjet_c;
    reco::CATopJetTagInfoRef                                            catopjet_r;
    reco::CATopJetTagInfoRefProd                                        catopjet_rp;
    reco::CATopJetTagInfoRefVector                                      catopjet_rv;
    edm::Wrapper<reco::CATopJetTagInfoCollection>                       catopjet_wc;
    edm::reftobase::Holder<reco::BaseTagInfo, reco::CATopJetTagInfoRef> rb_catopjet;
    edm::reftobase::RefHolder<reco::CATopJetTagInfoRef>                 rbh_catopjet; 
  }
}
