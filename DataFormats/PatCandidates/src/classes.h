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
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
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
    /*   With direct VectorHolder   */
  /*   RefToBaseVector<Candidate> from PATObjects, not yet provided. Useful?   */
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
  edm::reftobase::VectorHolder<reco::Candidate, pat::CompositeCandidateRefVector>	rb_cand_vh_p_cc;
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
  edm::reftobase::RefVectorHolder<pat::CompositeCandidateRefVector> rb_rvh_p_cc;

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
  std::pair<std::string, std::vector<float> > jcfcf;
  edm::Wrapper<std::pair<std::string, std::vector<float> > > w_jcfcf;
  std::vector<pat::JetCorrFactors::CorrectionFactor> v_jcfcf;
  edm::Wrapper<std::vector<pat::JetCorrFactors::CorrectionFactor> > w_v_jcfcf;
  std::vector<pat::JetCorrFactors> v_jcf;
  edm::Wrapper<std::vector<pat::JetCorrFactors> >  w_v_jcf;
  edm::ValueMap<pat::JetCorrFactors> vm_jcf;
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
  pat::L1Seed p_bs;
  pat::L1SeedCollection vp_bs;
  pat::L1SeedCollection::const_iterator vp_bs_ci;

  pat::TriggerConditionCollection v_p_tc;
  pat::TriggerConditionCollection::const_iterator v_p_tc_ci;
  edm::Wrapper<pat::TriggerConditionCollection> w_v_p_tc;
  pat::TriggerConditionRef r_p_tc;
  pat::TriggerConditionRefProd rp_p_tc;
  edm::Wrapper<pat::TriggerConditionRefProd> w_rp_p_tc;
  pat::TriggerConditionRefVector rv_p_tc;
  pat::TriggerConditionRefVectorIterator rv_p_tc_i;

  pat::TriggerAlgorithmCollection v_p_ta;
  pat::TriggerAlgorithmCollection::const_iterator v_p_ta_ci;
  edm::Wrapper<pat::TriggerAlgorithmCollection> w_v_p_ta;
  pat::TriggerAlgorithmRef r_p_ta;
  pat::TriggerAlgorithmRefProd rp_p_ta;
  edm::Wrapper<pat::TriggerAlgorithmRefProd> w_rp_p_ta;
  pat::TriggerAlgorithmRefVector rv_p_ta;
  pat::TriggerAlgorithmRefVectorIterator rv_p_ta_i;

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
