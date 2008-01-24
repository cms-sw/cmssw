
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Particle.h"

#include "DataFormats/PatCandidates/interface/StringMap.h"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"


// we need these typedefs, it won't work directly - NO IDEA WHY!!!
namespace pat {
  typedef edm::Ref<std::vector<pat::Electron> > ElectronRef;
  typedef edm::Ref<std::vector<pat::Muon> >     MuonRef;
  typedef edm::Ref<std::vector<pat::Tau> >      TauRef;
  typedef edm::Ref<std::vector<pat::Photon> >   PhotonRef;
  typedef edm::Ref<std::vector<pat::Jet> >      JetRef;
  typedef edm::Ref<std::vector<pat::MET> >      METRef;
  typedef edm::Ref<std::vector<pat::Particle> > ParticleRef;
}


namespace {
  namespace {

    std::pair<std::string,int32_t> smap0;
    std::vector<std::pair<std::string,int32_t> > smap1;
    StringMap smap;
    edm::Wrapper<StringMap> smap_w;

    std::pair<edm::RefToBase<reco::Candidate>,int32_t> hypo0;
    std::vector<std::pair<edm::RefToBase<reco::Candidate>,int32_t> > hypo1;
    pat::EventHypothesis hypot;
    std::vector<pat::EventHypothesis> hypots;
    edm::Wrapper<std::vector<pat::EventHypothesis> > hypots_w;

    

    // To check:
    // These don't belong here, and maybe they already exist in the meantime
    std::pair<int,float>  dummy0;
    std::pair<float,float>  dummy0_1;
    std::pair<std::string,float>  dummy0_2;
    std::vector<std::pair<int, float> >    dummy1;
    std::vector<std::vector<std::pair<int, float> >  >  dummy2;
    std::vector<std::pair<float, float> >                 v_p_dbl_dbl;
    std::pair<unsigned int, std::vector<unsigned int> >   p_uint_vint;
    std::vector<std::pair<std::string, float> >           v_p_str_dbl;
    std::vector<std::pair<unsigned int, float> >          v_p_uint_dbl;
    std::pair<unsigned int, float>                        p_uint_dbl;
    std::vector<std::pair<std::string, reco::JetTagRef> > v_p_str_jtr;
    std::pair<std::string, reco::JetTagRef>               p_str_jtr;

    pat::PATObject<pat::ElectronType>           po_el;
    pat::PATObject<pat::MuonType>               po_mu;
    pat::PATObject<pat::TauType>                po_tau;
    pat::PATObject<pat::PhotonType>             po_photon;
    pat::PATObject<pat::JetType>                po_jet;
    pat::PATObject<pat::METType>                po_met;
    pat::PATObject<pat::ParticleType>           po_part;
    pat::Lepton<pat::ElectronType>              tl_el;
    pat::Lepton<pat::MuonType>                  tl_mu;
    pat::Lepton<pat::TauType>                   tl_tau;
    pat::Electron                               el;
    pat::Muon                                   mu;
    pat::Tau                                    tau;
    pat::Photon                                 photon;
    pat::Jet                                    jet;
    pat::MET                                    met;
    pat::Particle                               part;
    std::vector<pat::Electron>                  v_el;
    std::vector<pat::Muon>                      v_mu;
    std::vector<pat::Tau>                       v_tau;
    std::vector<pat::Photon>                    v_photon;
    std::vector<pat::Jet>                       v_jet;
    std::vector<pat::MET>                       v_met;
    std::vector<pat::Particle>                  v_part;
    edm::Wrapper<std::vector<pat::Electron> >   w_v_el;
    edm::Wrapper<std::vector<pat::Muon> >       w_v_mu;
    edm::Wrapper<std::vector<pat::Tau> >        w_v_tau;
    edm::Wrapper<std::vector<pat::Photon> >     w_v_photon;
    edm::Wrapper<std::vector<pat::Jet> >        w_v_jet;
    edm::Wrapper<std::vector<pat::MET> >        w_v_met;
    edm::Wrapper<std::vector<pat::Particle> >   w_v_part;
    edm::Ref<std::vector<pat::Electron> >       r_el;
    edm::Ref<std::vector<pat::Muon> >           r_mu;
    edm::Ref<std::vector<pat::Tau> >            r_tau;
    edm::Ref<std::vector<pat::Photon> >         r_photon;
    edm::Ref<std::vector<pat::Jet> >            r_jet;
    edm::Ref<std::vector<pat::MET> >            r_met;
    edm::Ref<std::vector<pat::Particle> >       r_part;

  }
}
