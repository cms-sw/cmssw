#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"

#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

#include "DataFormats/PatCandidates/interface/StringMap.h"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"

// vvvv Needed to fix dictionaries missing in 169pre2
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
//#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
// ^^^^^ End

// we need these typedefs, it won't work directly - NO IDEA WHY!!!
namespace pat {
  typedef edm::Ref<std::vector<pat::Electron> > ElectronRef;
  typedef edm::Ref<std::vector<pat::Muon> >     MuonRef;
  typedef edm::Ref<std::vector<pat::Tau> >      TauRef;
  typedef edm::Ref<std::vector<pat::Photon> >   PhotonRef;
  typedef edm::Ref<std::vector<pat::Jet> >      JetRef;
  typedef edm::Ref<std::vector<pat::MET> >      METRef;
  typedef edm::Ref<std::vector<pat::Particle> > ParticleRef;
  typedef edm::Ref<std::vector<pat::Hemisphere> > HemisphereRef;

  typedef edm::Ref<std::vector<pat::ElectronType> > ElectronTypeRef;
  typedef edm::Ref<std::vector<pat::MuonType> >     MuonTypeRef;
  typedef edm::Ref<std::vector<pat::TauType> >      TauTypeRef;
  typedef edm::Ref<std::vector<pat::PhotonType> >   PhotonTypeRef;
  typedef edm::Ref<std::vector<pat::JetType> >      JetTypeRef;
  typedef edm::Ref<std::vector<pat::METType> >      METTypeRef;
  typedef edm::Ref<std::vector<pat::ParticleType> > ParticleTypeRef;
}


namespace {
  namespace {

    std::pair<std::string,int32_t> smap0;
    std::vector<std::pair<std::string,int32_t> > smap1;
    StringMap smap;
    edm::Wrapper<StringMap> smap_w;

    std::pair<std::string, edm::RefToBase<reco::Candidate> > hypo0;
    std::vector<std::pair<std::string, edm::RefToBase<reco::Candidate> > > hypo1;
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
    //    std::vector<std::pair<std::string, reco::JetTagRef> > v_p_str_jtr;
    //    std::pair<std::string, reco::JetTagRef>               p_str_jtr;

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
    pat::Hemisphere                             hemisphere;
    std::vector<pat::Electron>                  v_el;
    std::vector<pat::Muon>                      v_mu;
    std::vector<pat::Tau>                       v_tau;
    std::vector<pat::Photon>                    v_photon;
    std::vector<pat::Jet>                       v_jet;
    std::vector<pat::MET>                       v_met;
    std::vector<pat::Particle>                  v_part;
    std::vector<pat::Hemisphere>                v_hemi;
    edm::Wrapper<std::vector<pat::Electron> >   w_v_el;
    edm::Wrapper<std::vector<pat::Muon> >       w_v_mu;
    edm::Wrapper<std::vector<pat::Tau> >        w_v_tau;
    edm::Wrapper<std::vector<pat::Photon> >     w_v_photon;
    edm::Wrapper<std::vector<pat::Jet> >        w_v_jet;
    edm::Wrapper<std::vector<pat::MET> >        w_v_met;
    edm::Wrapper<std::vector<pat::Particle> >   w_v_part;
    edm::Wrapper<std::vector<pat::Hemisphere> >   w_v_hemi;
    edm::Ref<std::vector<pat::Electron> >       r_el;
    edm::Ref<std::vector<pat::Muon> >           r_mu;
    edm::Ref<std::vector<pat::Tau> >            r_tau;
    edm::Ref<std::vector<pat::Photon> >         r_photon;
    edm::Ref<std::vector<pat::Jet> >            r_jet;
    edm::Ref<std::vector<pat::MET> >            r_met;
    edm::Ref<std::vector<pat::Particle> >       r_part;
    edm::Ref<std::vector<pat::Hemisphere> >     r_hemi;

    pat::JetCorrFactors jcf;
    std::vector<pat::JetCorrFactors> v_jcf;
    edm::Wrapper<pat::JetCorrFactors> w_jcf;
    edm::ValueMap<pat::JetCorrFactors> vm_jcf;
    edm::Wrapper<edm::ValueMap<pat::JetCorrFactors> > wvm_jcf;

    //=========================================================
    //=== Dictionaries missing in 169pre2, we add them here ===
    //=========================================================
    edm::reftobase::RefHolder<reco::METRef> rb1a;
    edm::reftobase::RefHolder<reco::CaloMETRef> rb2a;
    edm::reftobase::RefHolder<reco::GenMETRef> rb3a;

    //    edm::Wrapper<edm::ValueMap<reco::JetTagRef> > rjtvm1; 

    edm::Wrapper<edm::Association<reco::GenJetCollection> > rgjc;
    
    edm::RefToBase<pat::ElectronType>  rbElectron;
    edm::reftobase::IndirectHolder<pat::ElectronType> rbihElectron;
    edm::reftobase::Holder<pat::ElectronType, pat::ElectronTypeRef> rbh1Electron;
    edm::reftobase::Holder<pat::ElectronType, pat::ElectronRef>     rbh2Electron;
    edm::reftobase::RefHolder<pat::ElectronRef> rhElectron;
    edm::RefToBase<pat::MuonType>  rbMuon;
    edm::reftobase::IndirectHolder<pat::MuonType> rbihMuon;
    edm::reftobase::Holder<pat::MuonType, pat::MuonTypeRef> rbh1Muon;
    edm::reftobase::Holder<pat::MuonType, pat::MuonRef>     rbh2Muon;
    edm::reftobase::RefHolder<pat::MuonRef> rhMuon;
    edm::RefToBase<pat::TauType>  rbTau;
    edm::reftobase::IndirectHolder<pat::TauType> rbihTau;
    edm::reftobase::Holder<pat::TauType, pat::TauTypeRef> rbh1Tau;
    edm::reftobase::Holder<pat::TauType, pat::TauRef>     rbh2Tau;
    edm::reftobase::RefHolder<pat::TauRef> rhTau;
    edm::RefToBase<pat::PhotonType>  rbPhoton;
    edm::reftobase::IndirectHolder<pat::PhotonType> rbihPhoton;
    edm::reftobase::Holder<pat::PhotonType, pat::PhotonTypeRef> rbh1Photon;
    edm::reftobase::Holder<pat::PhotonType, pat::PhotonRef>     rbh2Photon;
    edm::reftobase::RefHolder<pat::PhotonRef> rhPhoton;
    edm::RefToBase<pat::JetType>  rbJet;
    edm::reftobase::IndirectHolder<pat::JetType> rbihJet;
    edm::reftobase::Holder<pat::JetType, pat::JetTypeRef> rbh1Jet;
    edm::reftobase::Holder<pat::JetType, pat::JetRef>     rbh2Jet;
    edm::reftobase::RefHolder<pat::JetRef> rhJet;
    edm::RefToBase<pat::METType>  rbMET;
    edm::reftobase::IndirectHolder<pat::METType> rbihMET;
    edm::reftobase::Holder<pat::METType, pat::METTypeRef> rbh1MET;
    edm::reftobase::Holder<pat::METType, pat::METRef>     rbh2MET;
    edm::reftobase::RefHolder<pat::METRef> rhMET;
    edm::RefToBase<pat::ParticleType>  rbParticle;
    edm::reftobase::IndirectHolder<pat::ParticleType> rbihParticle;
    edm::reftobase::Holder<pat::ParticleType, pat::ParticleTypeRef> rbh1Particle;
    edm::reftobase::Holder<pat::ParticleType, pat::ParticleRef>     rbh2Particle;
    edm::reftobase::RefHolder<pat::ParticleRef> rhParticle;

    edm::Wrapper<edm::ValueMap<reco::TrackRefVector> > patJTA;
   
    std::vector<edm::Ref<std::vector<reco::SecondaryVertexTagInfo>,reco::SecondaryVertexTagInfo,edm::refhelper::FindUsingAdvance<std::vector<reco::SecondaryVertexTagInfo>,reco::SecondaryVertexTagInfo> > > rbh1btag ;
    std::vector<edm::Ref<std::vector<reco::SoftLeptonTagInfo>,reco::SoftLeptonTagInfo,edm::refhelper::FindUsingAdvance<std::vector<reco::SoftLeptonTagInfo>,reco::SoftLeptonTagInfo> > > rbh2btag; 

    std::vector<edm::Ref<std::vector<reco::TrackIPTagInfo>,reco::TrackIPTagInfo,edm::refhelper::FindUsingAdvance<std:: vector<reco::TrackIPTagInfo>,reco::TrackIPTagInfo> > > rbh3btag;
    std::vector<std::pair<pat::IsolationKeys,reco::IsoDeposit> > rbh4btag;

    
    std::pair<pat::IsolationKeys,reco::IsoDeposit> rbh4unknown;

    edm::Wrapper<edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > > wbti;

    }
}
