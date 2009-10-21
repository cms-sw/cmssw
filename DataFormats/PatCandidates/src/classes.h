#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/MHT.h"
#include "DataFormats/PatCandidates/interface/Particle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"

#include "DataFormats/PatCandidates/interface/StringMap.h"
#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/TriggerFilter.h"
#include "DataFormats/PatCandidates/interface/TriggerPath.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"

#include "DataFormats/PatCandidates/interface/Vertexing.h"

#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "DataFormats/PatCandidates/interface/CandKinResolution.h"

namespace {
  struct dictionary {

  /*   ==========================================================================================================================
              NON PAT Dataformats, except those for RefToBase/Ptr
       ==========================================================================================================================   */

  /*   To go into DataFormats/Candidate   */
  edm::Wrapper<edm::ValueMap<reco::CandidatePtr> >	            w_vm_cptr;
  std::vector<std::pair<std::string,edm::Ptr<reco::Candidate> > >   v_p_s_cptr;

  /*   To go into DataFormats/HepMCCandidate   */
  std::vector<reco::GenParticleRef>	v_gpr;

  /*   To go into DataFormats/JetReco   */
  edm::Wrapper<edm::Association<reco::GenJetCollection> >   w_a_gj;
  std::vector<reco::CaloJet::Specific>	                    v_cj_s;
  std::vector<reco::PFJet::Specific>	                    v_pj_s;

  /*   To go into DataFormats/BTauReco   */
  edm::Wrapper<edm::ValueMap<edm::Ptr<reco::BaseTagInfo> > >	          w_vm_ptr_bti;
  std::vector<reco::BaseTagInfo*>                                         pv_bti;
  edm::OwnVector<reco::BaseTagInfo, edm::ClonePolicy<reco::BaseTagInfo> > ov_bti;
  edm::Ptr<reco::BaseTagInfo>                                             ptr_bti;

  /*   To go into DataFormats/TrackReco   */
  edm::Wrapper<edm::ValueMap<reco::TrackRefVector> >	 w_vm_trv;


   /*   ==========================================================================================================================
              NON PAT Dataformats: RefToBase for Candidates
                                   Needed to make RefToBase<Candidate> from AOD objects: 
                                        GsfElectron, Muon, Photon, BaseTau, MET, CaloMet, GenMET, 
                                        CaloTau, PFTau, CaloJet, PFJet, BasicJet, GenJet
                                   No longer needed by PAT, but needed by core PhysicsTools.
       ==========================================================================================================================   */
  /*  
  edm::reftobase::RefHolder<reco::BasicJetRef>	rb_rh_bj;

  edm::reftobase::Holder<reco::Candidate, reco::PFTauRef>   rb_cand_h_pft; 
  edm::reftobase::RefHolder<reco::BaseTauRef>	            rb_rh_pft;
  edm::reftobase::RefHolder<reco::CaloTauRef>	            rb_rh_ct;
  edm::reftobase::RefHolder<reco::PFTauRef>	            rb_rh_bt;
  */

   /*   ==========================================================================================================================
              NON PAT Dataformats: RefToBaseVector for Candidates
                                   Needed to make RefToBaseVector<Candidate> from AOD objects: 
                                        GsfElectron, Muon, Photon, BaseTau, MET, CaloMet, GenMET, 
                                        CaloTau, PFTau, CaloJet, PFJet, BasicJet, GenJet
                                   No longer needed by PAT, but probably needed by core PhysicsTools.
       ==========================================================================================================================   */
  /*  

  edm::reftobase::RefVectorHolder<reco::CaloJetRefVector>                 rb_rvh_cj;
  edm::reftobase::RefVectorHolder<reco::PFJetRefVector>	                  rb_rvh_pfj;
  edm::reftobase::RefVectorHolder<reco::BasicJetRefVector>                rb_rvh_bj;
  edm::reftobase::VectorHolder<reco::Candidate, reco::CaloJetRefVector>   rb_cand_vh_cj;
  edm::reftobase::VectorHolder<reco::Candidate, reco::PFJetRefVector>	  rb_cand_vh_pfj;
  edm::reftobase::VectorHolder<reco::Candidate, reco::BasicJetRefVector>  rb_cand_vh_bj;

  edm::reftobase::RefVectorHolder<reco::BaseTauRefVector>                 rb_rvh_bt;
  edm::reftobase::RefVectorHolder<reco::CaloTauRefVector>                 rb_rvh_ct;
  edm::reftobase::RefVectorHolder<reco::PFTauRefVector>	                  rb_rvh_pft;
  edm::reftobase::VectorHolder<reco::Candidate, reco::BaseTauRefVector>   rb_cand_vh_bt;
  edm::reftobase::VectorHolder<reco::Candidate, reco::CaloTauRefVector>   rb_cand_vh_ct;
  edm::reftobase::VectorHolder<reco::Candidate, reco::PFTauRefVector>	  rb_cand_vh_pft;

  edm::reftobase::RefVectorHolder<reco::METRefVector>	                rb_rvh_m;
  edm::reftobase::RefVectorHolder<reco::CaloMETRefVector>               rb_rvh_cm;
  edm::reftobase::RefVectorHolder<reco::GenMETRefVector>                rb_rvh_gm;
  edm::reftobase::VectorHolder<reco::Candidate, reco::METRefVector>	rb_cand_vh_m;
  edm::reftobase::VectorHolder<reco::Candidate, reco::CaloMETRefVector> rb_cand_vh_cm;
  edm::reftobase::VectorHolder<reco::Candidate, reco::GenMETRefVector>  rb_cand_vh_gm;
  */


  /*   ==========================================================================================================================
              NON PAT Dataformats: RefToBase for AOD Types (only the ones missing from 2.1.9)
                                   Should be elsewhere. Also, no longer needed by PAT.
                                   They allow to make RefToBase<T> and RefToBaseVector<T> 
                                   for each T = GsfElectron, Muon, Photon, Jet, BaseTau, MET
                                   filled from all concrete types (all T above except Jet, 
                                   plus CaloMet, GenMET, CaloTau, PFTau, CaloJet, PFJet, BasicJet).
                                   This does not include those alrady needed to make RefToBase(Vector)<Candidate>
       ==========================================================================================================================   */
  /*  
  edm::reftobase::Holder<reco::GsfElectron, reco::GsfElectronRef>               rb_e_h_e;
  edm::reftobase::IndirectVectorHolder<reco::GsfElectron>	                rb_e_ivh;
  edm::reftobase::VectorHolder<reco::GsfElectron, reco::GsfElectronRefVector>	rb_e_vh_e;

  edm::reftobase::Holder<reco::Photon, reco::PhotonRef>	            rb_ph_h_ph;
  edm::reftobase::IndirectVectorHolder<reco::Photon>	            rb_ph_ivh;
  edm::reftobase::VectorHolder<reco::Photon, reco::PhotonRefVector> rb_ph_vh_ph;

  edm::reftobase::IndirectVectorHolder<reco::Muon>	         rb_mu_ivh;
  edm::reftobase::Holder<reco::Muon, reco::MuonRef>	         rb_mu_h_mu;
  edm::reftobase::VectorHolder<reco::Muon, reco::MuonRefVector>	 rb_mu_vh_mu;

  edm::reftobase::IndirectVectorHolder<reco::Jet>	            rb_j_ivh;
  edm::reftobase::Holder<reco::Jet, reco::BasicJetRef>              rb_j_v_bj;
  edm::reftobase::VectorHolder<reco::Jet, reco::CaloJetRefVector>   rb_j_vh_cj;
  edm::reftobase::VectorHolder<reco::Jet, reco::PFJetRefVector>	    rb_j_vh_pfj;
  edm::reftobase::VectorHolder<reco::Jet, reco::BasicJetRefVector>  rb_j_vh_bj;
  edm::reftobase::VectorHolder<reco::Jet, reco::GenJetRefVector>    rb_j_vh_gj;
  
  edm::RefToBase<reco::BaseTau>	                                 rb_t;
  edm::RefToBaseVector<reco::BaseTau>	                         rb_tv;
  edm::reftobase::IndirectHolder<reco::BaseTau>	                 rb_t_ih;
  edm::reftobase::IndirectVectorHolder<reco::BaseTau>	         rb_t_ivh;
  edm::reftobase::Holder<reco::BaseTau, reco::BaseTauRef>	 rb_t_h_bt;
  edm::reftobase::VectorHolder<reco::BaseTau, reco::BaseTauRef>	 rb_t_vh_bt;
  edm::reftobase::VectorHolder<reco::BaseTau, reco::CaloTauRef>	 rb_t_vh_ct;
  edm::reftobase::VectorHolder<reco::BaseTau, reco::PFTauRef>	 rb_t_vh_pft;

  edm::RefToBase<reco::MET>	                                    rb_m;
  edm::RefToBaseVector<reco::MET>	                            rb_mv;
  edm::reftobase::IndirectHolder<reco::MET>	                    rb_m_ih;
  edm::reftobase::IndirectVectorHolder<reco::MET>	            rb_m_ivh;
  edm::reftobase::Holder<reco::MET, reco::METRef>	            rb_m_h_m;
  edm::reftobase::Holder<reco::MET, reco::CaloMETRef>	            rb_m_h_cm;
  edm::reftobase::Holder<reco::MET, reco::GenMETRef>	            rb_m_h_gm;
  edm::reftobase::VectorHolder<reco::MET, reco::METRefVector>	    rb_m_vh_m;
  edm::reftobase::VectorHolder<reco::MET, reco::CaloMETRefVector>   rb_m_vh_cm;
  edm::reftobase::VectorHolder<reco::MET, reco::GenMETRefVector>    rb_m_vh_gm;
    */

  /*   ==========================================================================================================================
              NON PAT Dataformats: Ptr for AOD Types
                                   Should be elsewhere.
                                   Needed by PAT after reshuffling, at least for Electrons and Photons
       ==========================================================================================================================   */
  edm::Ptr<reco::GsfElectron>	     ptr_e;
  edm::PtrVector<reco::GsfElectron>  ptrv_e;

  edm::Ptr<reco::BaseTau>	 ptr_t;
  edm::PtrVector<reco::BaseTau>	 ptrv_t;

  edm::Ptr<reco::Photon>	 ptr_ph;
  edm::PtrVector<reco::Photon>	 ptrv_ph;

  edm::Ptr<reco::Jet>	     ptr_j;
  edm::PtrVector<reco::Jet>  ptrv_j;

  edm::Ptr<reco::MET>	     ptr_m;
  edm::PtrVector<reco::MET>  ptrv_m;

  /*   ==========================================================================================================================
              PAT Dataformats: PatObjects
       ==========================================================================================================================   */
  /*   PAT Object Collection Iterators   */
  std::vector<pat::Electron>::const_iterator	    v_p_e_ci;
  std::vector<pat::Muon>::const_iterator	    v_p_mu_ci;
  std::vector<pat::Tau>::const_iterator	            v_p_t_ci;
  std::vector<pat::Photon>::const_iterator	    v_p_ph_ci;
  std::vector<pat::Jet>::const_iterator	            v_p_j_ci;
  std::vector<pat::MET>::const_iterator	            v_p_m_ci;
  std::vector<pat::MHT>::const_iterator	            v_p_mht_ci;
  std::vector<pat::Particle>::const_iterator	    v_p_p_ci;
  std::vector<pat::CompositeCandidate>::const_iterator	v_p_cc_ci;
  std::vector<pat::PFParticle>::const_iterator	    v_p_pfp_ci;
  std::vector<pat::GenericParticle>::const_iterator v_p_gp_ci;
  std::vector<pat::Hemisphere>::const_iterator	    v_p_h_ci;

  /*   PAT Object Collection Wrappers   */
  edm::Wrapper<std::vector<pat::Electron> >	    w_v_p_e;
  edm::Wrapper<std::vector<pat::Muon> >	            w_v_p_mu;
  edm::Wrapper<std::vector<pat::Tau> >	            w_v_p_t;
  edm::Wrapper<std::vector<pat::Photon> >	    w_v_p_ph;
  edm::Wrapper<std::vector<pat::Jet> >	            w_v_p_j;
  edm::Wrapper<std::vector<pat::MET> >	            w_v_p_m;
  edm::Wrapper<std::vector<pat::MHT> >	            w_v_p_mht;
  edm::Wrapper<std::vector<pat::Particle> >	    w_v_p_p;
  edm::Wrapper<std::vector<pat::CompositeCandidate> > w_v_cc_p;
  edm::Wrapper<std::vector<pat::PFParticle> >	    w_v_p_pfp;
  edm::Wrapper<std::vector<pat::GenericParticle> >  w_v_p_gp;
  edm::Wrapper<std::vector<pat::Hemisphere> >	    w_v_p_h;

  /*   PAT Object References   */
  pat::ElectronRef	    p_r_e;
  pat::MuonRef	            p_r_mu;
  pat::TauRef	            p_r_t;
  pat::PhotonRef	    p_r_ph;
  pat::JetRef	            p_r_j;
  pat::METRef	            p_r_m;
  pat::ParticleRef	    p_r_p;
  pat::CompositeCandidateRef	    p_r_cc;
  pat::PFParticleRef	    p_r_pgp;
  pat::GenericParticleRef   p_r_gp;
  pat::HemisphereRef	    p_r_h;

  /*   PAT Object Ref Vector Wrappers   */
  edm::Wrapper<pat::ElectronRefVector>	        p_rv_e;
  edm::Wrapper<pat::MuonRefVector>	        p_rv_mu;
  edm::Wrapper<pat::TauRefVector>	        p_rv_t;
  edm::Wrapper<pat::PhotonRefVector>	        p_rv_ph;
  edm::Wrapper<pat::JetRefVector>	        p_rv_j;
  edm::Wrapper<pat::METRefVector>	        p_rv_m;
  edm::Wrapper<pat::ParticleRefVector>	        p_rv_p;
  edm::Wrapper<pat::CompositeCandidateRefVector> p_rv_cc;
  edm::Wrapper<pat::PFParticleRefVector>	p_rv_pgp;
  edm::Wrapper<pat::GenericParticleRefVector>   p_rv_gp;
  edm::Wrapper<pat::HemisphereRefVector>	p_rv_h;

  /*   RefToBase<Candidate> from PATObjects   */
    /*   With direct Holder   */
  edm::reftobase::Holder<reco::Candidate, pat::ElectronRef>	 	rb_cand_h_p_e;
  edm::reftobase::Holder<reco::Candidate, pat::MuonRef>	         	rb_cand_h_p_mu;
  edm::reftobase::Holder<reco::Candidate, pat::TauRef>	         	rb_cand_h_p_t;
  edm::reftobase::Holder<reco::Candidate, pat::PhotonRef>	 	rb_cand_h_p_ph;
  edm::reftobase::Holder<reco::Candidate, pat::JetRef>	         	rb_cand_h_p_j;
  edm::reftobase::Holder<reco::Candidate, pat::METRef>	         	rb_cand_h_p_m;
  edm::reftobase::Holder<reco::Candidate, pat::ParticleRef>	 	rb_cand_h_p_p;
  edm::reftobase::Holder<reco::Candidate, pat::CompositeCandidateRef>	rb_cand_h_p_cc;
  edm::reftobase::Holder<reco::Candidate, pat::PFParticleRef>	 	rb_cand_h_p_pfp;
  edm::reftobase::Holder<reco::Candidate, pat::GenericParticleRef>	rb_cand_h_p_gp;
    /*   With indirect holder (RefHolder)   */
  edm::reftobase::RefHolder<pat::ElectronRef>	 	rb_rh_p_e;
  edm::reftobase::RefHolder<pat::MuonRef>	 	rb_rh_p_mu;
  edm::reftobase::RefHolder<pat::TauRef>	 	rb_rh_p_t;
  edm::reftobase::RefHolder<pat::PhotonRef>	 	rb_rh_p_ph;
  edm::reftobase::RefHolder<pat::JetRef>	 	rb_rh_p_j;
  edm::reftobase::RefHolder<pat::METRef>	 	rb_rh_p_m;
  edm::reftobase::RefHolder<pat::ParticleRef>	 	rb_rh_p_p;
  edm::reftobase::RefHolder<pat::CompositeCandidateRef>	rb_rh_p_cc;
  edm::reftobase::RefHolder<pat::PFParticleRef>	 	rb_rh_p_pfp;
  edm::reftobase::RefHolder<pat::GenericParticleRef>    rb_rh_p_gp;
  /*   RefToBaseVector<Candidate> from PATObjects, not yet provided. Useful?   */
    /*   With direct VectorHolder   */
  /*  
  edm::reftobase::VectorHolder<reco::Candidate, pat::ElectronRefVector>	        rb_cand_vh_p_e;
  edm::reftobase::VectorHolder<reco::Candidate, pat::MuonRefVector>	        rb_cand_vh_p_mu;
  edm::reftobase::VectorHolder<reco::Candidate, pat::TauRefVector>	        rb_cand_vh_p_t;
  edm::reftobase::VectorHolder<reco::Candidate, pat::PhotonRefVector>	        rb_cand_vh_p_ph;
  edm::reftobase::VectorHolder<reco::Candidate, pat::JetRefVector>	        rb_cand_vh_p_j;
  edm::reftobase::VectorHolder<reco::Candidate, pat::METRefVector>	        rb_cand_vh_p_m;
  edm::reftobase::VectorHolder<reco::Candidate, pat::ParticleRefVector>	        rb_cand_vh_p_p;
  edm::reftobase::VectorHolder<reco::Candidate, pat::PFParticleRefVector>	rb_cand_vh_p_pfp;
  edm::reftobase::VectorHolder<reco::Candidate, pat::GenericParticleRefVector>	rb_cand_vh_p_gp;
  */
    /*   With indirect holder (RefVectorHolder)   */
  /*  
  edm::reftobase::RefVectorHolder<pat::ElectronRefVector>	 rb_rvh_p_e;
  edm::reftobase::RefVectorHolder<pat::MuonRefVector>	         rb_rvh_p_mu;
  edm::reftobase::RefVectorHolder<pat::TauRefVector>	         rb_rvh_p_t;
  edm::reftobase::RefVectorHolder<pat::PhotonRefVector>	         rb_rvh_p_ph;
  edm::reftobase::RefVectorHolder<pat::JetRefVector>	         rb_rvh_p_j;
  edm::reftobase::RefVectorHolder<pat::METRefVector>	         rb_rvh_p_m;
  edm::reftobase::RefVectorHolder<pat::ParticleRefVector>	 rb_rvh_p_p;
  edm::reftobase::RefVectorHolder<pat::PFParticleRefVector>	 rb_rvh_p_pfp;
  edm::reftobase::RefVectorHolder<pat::GenericParticleRefVector> rb_rvh_p_gp;
  */

  /*   RefToBase<AODType> from PATObjects. In addition to the ones for Candidate    */
  /*  
  edm::reftobase::Holder<reco::GsfElectron, pat::ElectronRef>	rb_e_h_p_e;
  edm::reftobase::Holder<reco::Muon, pat::MuonRef>	        rb_mu_h_p_mu;
  edm::reftobase::Holder<reco::BaseTau, pat::TauRef>	        rb_t_h_p_t;
  edm::reftobase::Holder<reco::Photon, pat::PhotonRef>		rb_ph_h_p_ph;
  edm::reftobase::Holder<reco::Jet, pat::JetRef>	        rb_j_h_p_j;
  edm::reftobase::Holder<reco::MET, pat::METRef>	        rb_m_h_p_m;

  edm::reftobase::VectorHolder<reco::GsfElectron, pat::ElectronRefVector>   rb_e_vh_p_e;
  edm::reftobase::VectorHolder<reco::Muon, pat::MuonRefVector>		    rb_mu_vh_p_mu;
  edm::reftobase::VectorHolder<reco::BaseTau, pat::TauRefVector> 	    rb_t_vh_p_t;
  edm::reftobase::VectorHolder<reco::Photon, pat::PhotonRefVector>	    rb_ph_vh_p_ph;
  edm::reftobase::VectorHolder<reco::Jet, pat::JetRefVector> 		    rb_j_vh_p_j;
  edm::reftobase::VectorHolder<reco::MET, pat::METRefVector> 		    rb_m_vh_p_m;
  */

  /*   ==========================================================================================================================
              PAT Dataformats beyond PatObjects
       ==========================================================================================================================   */
  edm::Wrapper<edm::ValueMap<pat::JetCorrFactors> >  w_vm_jcf;

  edm::Wrapper<StringMap>   w_sm;

  edm::Wrapper<edm::ValueMap<pat::VertexAssociation> >	 w_vm_va;

  edm::Wrapper<std::vector<pat::EventHypothesis> >	 w_v_eh;

  pat::TriggerObjectCollection v_p_to;
  pat::TriggerObjectCollection::const_iterator v_p_to_ci;
  edm::Wrapper<pat::TriggerObjectCollection> w_v_p_to;
  pat::TriggerObjectRef r_p_to;
  std::pair< std::string, pat::TriggerObjectRef > p_r_p_to;
  pat::TriggerObjectMatchMap m_r_p_to;
  pat::TriggerObjectRefProd rp_p_to;
  edm::Wrapper<pat::TriggerObjectRefProd> w_rp_p_to;
  pat::TriggerObjectRefVector rv_p_to;
  pat::TriggerObjectRefVectorIterator rv_p_to_i;
  pat::TriggerObjectMatch a_p_to;
  edm::reftobase::Holder<reco::Candidate, pat::TriggerObjectRef> h_p_to;
  edm::reftobase::RefHolder<pat::TriggerObjectRef> rh_p_to;
//   edm::reftobase::VectorHolder<reco::Candidate, pat::TriggerObjectRefVector> vh_p_to;
//   edm::reftobase::RefVectorHolder<pat::TriggerObjectRefVector> rvh_p_to;
  edm::Wrapper<pat::TriggerObjectMatch> w_a_p_to;
  pat::TriggerObjectMatchRefProd rp_a_p_to;
  std::pair< std::string, pat::TriggerObjectMatchRefProd > p_rp_a_p_to;
  pat::TriggerObjectMatchContainer m_rp_a_p_to;
  pat::TriggerObjectMatchContainer::const_iterator m_rp_a_p_to_ci;
  edm::Wrapper<pat::TriggerObjectMatchContainer> w_m_rp_a_p_to;

  pat::TriggerObjectStandAloneCollection v_p_tosa;
  pat::TriggerObjectStandAloneCollection::const_iterator v_p_tosa_ci;
  edm::Wrapper<pat::TriggerObjectStandAloneCollection> w_v_p_tosa;
  pat::TriggerObjectStandAloneRef r_p_tosa;
  pat::TriggerObjectStandAloneRefProd rp_p_tosa;
  edm::Wrapper<pat::TriggerObjectStandAloneRefProd> w_rp_p_tosa;
  pat::TriggerObjectStandAloneRefVector rv_p_tosa;
  pat::TriggerObjectStandAloneRefVectorIterator rv_p_tosa_i;
  pat::TriggerObjectStandAloneMatch a_p_tosa;
  edm::reftobase::Holder<reco::Candidate, pat::TriggerObjectStandAloneRef> h_p_tosa;
  edm::reftobase::RefHolder<pat::TriggerObjectStandAloneRef> rh_p_tosa;
//   edm::reftobase::VectorHolder<reco::Candidate, pat::TriggerObjectStandAloneRefVector> vh_p_tosa;
//   edm::reftobase::RefVectorHolder<pat::TriggerObjectStandAloneRefVector> rvh_p_tosa;
  edm::Wrapper<pat::TriggerObjectStandAloneMatch> w_a_p_tosa;

  pat::TriggerFilterCollection v_p_tf;
  pat::TriggerFilterCollection::const_iterator v_p_tf_ci;
  edm::Wrapper<pat::TriggerFilterCollection> w_v_p_tf;
  pat::TriggerFilterRef r_p_tf;
  pat::TriggerFilterRefProd rp_p_tf;
  edm::Wrapper<pat::TriggerFilterRefProd> w_rp_p_tf;
  pat::TriggerFilterRefVector rv_p_tf;
  pat::TriggerFilterRefVectorIterator rv_p_tf_i;

  pat::TriggerPathCollection v_p_tp;
  pat::TriggerPathCollection::const_iterator v_p_tp_ci;
  edm::Wrapper<pat::TriggerPathCollection> w_v_p_tp;
  pat::TriggerPathRef r_p_tp;
  pat::TriggerPathRefProd rp_p_tp;
  edm::Wrapper<pat::TriggerPathRefProd> w_rp_p_tp;
  pat::TriggerPathRefVector rv_p_tp;
  pat::TriggerPathRefVectorIterator rv_p_tp_i;
  
  edm::Wrapper<pat::TriggerEvent> w_p_te;

  std::vector<std::pair<pat::IsolationKeys,reco::IsoDeposit> >	 v_p_ik_id;

  /*   UserData: Core   */
  pat::UserDataCollection	         ov_p_ud;
  /*   UserData: Standalone UserData in the event. Needed?   */
  edm::Wrapper<pat::UserDataCollection>	 w_ov_p_ud;
  edm::Wrapper<edm::ValueMap<edm::Ptr<pat::UserData> > > w_vm_ptr_p_ud;
  edm::Ptr<pat::UserData> yadda_pat_ptr_userdata;
  /*   UserData: a few holders   */
  pat::UserHolder<math::XYZVector>	         p_udh_v3d;
  pat::UserHolder<math::XYZPoint>	         p_udh_p3d;
  pat::UserHolder<math::XYZTLorentzVector>	 p_udh_lv;
  pat::UserHolder<math::PtEtaPhiMLorentzVector>	 p_udh_plv;
  pat::UserHolder<AlgebraicSymMatrix22>          p_udh_smat_22;
  pat::UserHolder<AlgebraicSymMatrix33>          p_udh_smat_33;
  pat::UserHolder<AlgebraicSymMatrix44>          p_udh_smat_44;
  pat::UserHolder<AlgebraicSymMatrix55>          p_udh_smat_55;
  pat::UserHolder<AlgebraicVector2>              p_udh_vec_2;
  pat::UserHolder<AlgebraicVector3>              p_udh_vec_3;
  pat::UserHolder<AlgebraicVector4>              p_udh_vec_4;
  pat::UserHolder<AlgebraicVector5>              p_udh_vec_5;
  pat::UserHolder<reco::Track>                   p_udh_tk;
  pat::UserHolder<reco::Vertex>                  p_udh_vtx;


  edm::Wrapper<edm::ValueMap<pat::LookupTableRecord> >	 w_vm_p_lutr;

  pat::CandKinResolution ckr;
  std::vector<pat::CandKinResolution>  v_ckr;
  pat::CandKinResolutionValueMap vm_ckr;
  edm::Wrapper<pat::CandKinResolutionValueMap> w_vm_ckr;

  };

}
