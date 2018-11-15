#ifndef RecoTauTag_RecoTau_PFRecoTauClusterVariables_h
#define RecoTauTag_RecoTau_PFRecoTauClusterVariables_h

/** \class PFRecoTauClusterVariables
 *                                                                                                                                                                                   
 * A bunch of functions to return cluster variables used for the MVA based tau ID discrimation.
 * To allow the MVA based tau discrimination to be aplicable on miniAOD in addition to AOD 
 * several of these functions need to be overloaded.  
 *                                                                                                                                                                                   
 * \author Aruna Nayak, DESY
 *                                                                                                                                                                                   
 */


#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco { namespace tau {
  /// return chi2 of the leading track ==> deprecated? <==
  float lead_track_chi2(const reco::PFTau& tau);
  /// return ratio of energy in ECAL over sum of energy in ECAL and HCAL
  float eratio(const reco::PFTau& tau);
  float eratio(const pat::Tau& tau);
  /// return sum of pt weighted values of distance to tau candidate for all pf photon candidates, 
  /// which are associated to signal; depending on var the distance is in 0=:dr, 1=:deta, 2=:dphi 
  float pt_weighted_dx(const reco::PFTau& tau, int mode = 0, int var = 0, int decaymode = -1);
  float pt_weighted_dx(const pat::Tau& tau, int mode = 0, int var = 0, int decaymode = -1);
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_dr_signal(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, 0, 0, dm);
  }
  inline float pt_weighted_dr_signal(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, 0, 0, dm);
  }
  /// return sum of pt weighted values of deta relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_deta_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
  }
  inline float pt_weighted_deta_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
  }
  /// return sum of pt weighted values of dphi relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_dphi_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
  }
  inline float pt_weighted_dphi_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
  }  
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are inside an isolation conde but not associated to signal
  inline float pt_weighted_dr_iso(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, 2, 0, dm);
  }
  inline float pt_weighted_dr_iso(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, 2, 0, dm);
  }
  /// return total number of pf photon candidates with pT>500 MeV, which are associated to signal
  unsigned int n_photons_total(const reco::PFTau& tau);
  unsigned int n_photons_total(const pat::Tau& tau);
  
  enum {kOldDMwoLT, kOldDMwLT, kNewDMwoLT, kNewDMwLT, kDBoldDMwLT, kDBnewDMwLT, kPWoldDMwLT, kPWnewDMwLT, 
        kDBoldDMwLTwGJ, kDBnewDMwLTwGJ};
  bool fillIsoMVARun2Inputs(float* mvaInput, const pat::Tau& tau, int mvaOpt, const std::string& nameCharged,
                            const std::string& nameNeutral, const std::string& namePu, 
                            const std::string& nameOutside, const std::string& nameFootprint);
}} // namespaces

#endif
