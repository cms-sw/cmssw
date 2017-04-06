#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/FwdPtr.h"

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
#include "DataFormats/PatCandidates/interface/Conversion.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

namespace DataFormats_PatCandidates {
  struct dictionaryobjects {

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
  std::vector<pat::Conversion>::const_iterator      v_p_c_ci;
  std::vector<pat::PackedCandidate>::const_iterator      v_p_pc_ci;
  std::vector<pat::PackedGenParticle>::const_iterator      v_p_pgc_ci;

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
  edm::Wrapper<std::vector<pat::Conversion> >       w_v_p_c;
  edm::Wrapper<std::vector<pat::PackedCandidate> >       w_v_pc_c;
  edm::Wrapper<std::vector<pat::PackedGenParticle> >       w_v_pgc_c;

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
  pat::ConversionRef        p_r_c;
  pat::PackedCandidateRef        p_r_pc;
  pat::PackedGenParticleRef        p_r_pcg;

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
  edm::Wrapper<pat::ConversionRefVector>        p_rv_c;
  edm::Wrapper<pat::PackedCandidateRefVector>        p_rv_pc;
  edm::Wrapper<pat::PackedGenParticleRefVector>        p_rv_pcg;

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
  edm::reftobase::Holder<reco::Candidate, pat::ConversionRef>           rb_cand_h_p_c;
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
  edm::reftobase::RefHolder<pat::ConversionRef>         rb_rh_p_c;
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

  edm::Ptr<pat::Jet> ptr_Jet;
  edm::Ptr<pat::MET> ptr_MET;
  edm::Ptr<pat::Electron> ptr_Electron;
  edm::Ptr<pat::Conversion> ptr_Conversion;
  edm::Ptr<pat::Muon> ptr_Muon;
  edm::Ptr<pat::Tau> ptr_Tau;

  edm::FwdPtr<pat::PackedCandidate> fwdptr_pc;
  edm::Wrapper< edm::FwdPtr<pat::PackedCandidate> > w_fwdptr_pc;
  std::vector< edm::FwdPtr<pat::PackedCandidate> > v_fwdptr_pc;
  edm::Wrapper< std::vector< edm::FwdPtr<pat::PackedCandidate> > > wv_fwdptr_pc;

  edm::Wrapper<edm::Association<pat::PackedCandidateCollection > > w_asso_pc;
  edm::Wrapper<edm::Association<reco::PFCandidateCollection > >    w_asso_pfc;
  edm::Wrapper<edm::Association<std::vector<pat::PackedGenParticle> > > asso_pgp;


  std::vector< edm::Ptr<pat::Jet> > vptr_jet;
  std::vector< std::vector< edm::Ptr<pat::Jet> > > vvptr_jet;
  edm::Wrapper< std::vector< edm::Ptr<pat::Jet> > > wvptr_jet; 
  edm::Wrapper< std::vector< std::vector< edm::Ptr<pat::Jet> > > > wvvptr_jet;


  };

}
