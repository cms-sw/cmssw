#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiMassSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/JetRejObs.h"

#include "DataFormats/Common/interface/Wrapper.h"


namespace {
  namespace {
   std::pair<int,double>  dummy0;
   std::vector<std::pair<int, double> >    dummy1;
   std::vector<std::vector<std::pair<int, double> >  >  dummy2;
   typedef edm::Ref<std::vector<JetRejObs> > JetRejObsRef;

 
    JetRejObs                                  jro;
    std::vector<JetRejObs>                     v_jro;
    edm::Wrapper<std::vector<JetRejObs> >      w_v_jro;
    edm::Ref<std::vector<JetRejObs> >          r_jro;


    std::vector<std::pair<double, double> >               v_p_dbl_dbl;
    std::pair<unsigned int, std::vector<unsigned int> >   p_uint_vint;
    std::vector<std::pair<std::string, double> >          v_p_str_dbl;
    std::vector<std::pair<unsigned int, double> >         v_p_uint_dbl;
    std::pair<unsigned int, double>                       p_uint_dbl;
    std::vector<std::pair<std::string, reco::JetTagRef> > v_p_str_jtr;
    std::pair<std::string, reco::JetTagRef>               p_str_jtr;

    // we need these typedefs, it won't work directly - NO IDEA WHY!!!
    typedef edm::Ref<std::vector<TopElectron> > TopElectronRef;
    typedef edm::Ref<std::vector<TopMuon> >     TopMuonRef;
    typedef edm::Ref<std::vector<TopJet> >      TopJetRef;
    typedef edm::Ref<std::vector<TopMET> >      TopMETRef;
    typedef edm::Ref<std::vector<TopParticle> > TopParticleRef;

    TopObject<TopElectronType>                            to_el;
    TopObject<TopMuonType>                                to_mu;
    TopObject<TopJetType>                                 to_jet;
    TopObject<TopMETType>                                 to_met;
    TopObject<TopParticleType>                            to_part;
    TopElectron                                           el; 
    TopMuon                                               mu; 
    TopJet                                                jet;
    TopMET                                                met;
    TopParticle                                           part;
    std::vector<TopElectron>                              v_el;
    std::vector<TopMuon>                                  v_mu;
    std::vector<TopJet>                                   v_jet;
    std::vector<TopMET>                                   v_met;
    std::vector<TopParticle>                              v_part;
    edm::Wrapper<std::vector<TopElectron> >               w_v_el;
    edm::Wrapper<std::vector<TopMuon> >                   w_v_mu;
    edm::Wrapper<std::vector<TopJet> >                    w_v_jet;
    edm::Wrapper<std::vector<TopMET> >                    w_v_met;
    edm::Wrapper<std::vector<TopParticle> >               w_v_part;
    edm::Ref<std::vector<TopElectron> >                   r_el;
    edm::Ref<std::vector<TopMuon> >                       r_mu;
    edm::Ref<std::vector<TopJet> >                        r_jet;
    edm::Ref<std::vector<TopMET> >                        r_met;
    edm::Ref<std::vector<TopParticle> >                   r_part;

    TtGenEvent                                            ttgen;
    StGenEvent                                            stgen;
    TopGenEvent                                           topgen;
    edm::Wrapper<TtGenEvent>                              w_ttgen;
    edm::Wrapper<StGenEvent>                              w_stgen;
    edm::Wrapper<TopGenEvent>                             w_topgen;
    edm::RefProd<TtGenEvent>                              rp_ttgen;
    edm::RefProd<StGenEvent>                              rp_stgen;
    edm::RefProd<TopGenEvent>                             rp_topgen;
    std::vector<const reco::Candidate*>                   v_candvec;
    edm::Wrapper<std::vector<const reco::Candidate*> >    w_v_candvec;

    TtDilepEvtSolution                                    ttdilep;
    TtSemiEvtSolution                                     ttsemi;
    TtSemiMassSolution              	                  ttsemimass;
    StEvtSolution                                         st;
    std::vector<TtDilepEvtSolution>                       v_ttdilep;
    std::vector<TtSemiEvtSolution>                        v_ttsemi;
    std::vector<StEvtSolution>                            v_st;
    edm::Wrapper<std::vector<TtDilepEvtSolution> >        w_v_ttdilep;
    edm::Wrapper<std::vector<TtSemiEvtSolution> >         w_v_ttsemi;
    edm::Wrapper<std::vector<StEvtSolution> >             w_v_st;
    
  }
}
