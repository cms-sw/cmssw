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

namespace {
  template<class TauType, class PhotonVectorType>
  PhotonVectorType getGammas(const TauType& tau, bool signal);

  /// return pf photon candidates that are associated to signal
  template<>
  const std::vector<reco::PFCandidatePtr>& getGammas<reco::PFTau, const std::vector<reco::PFCandidatePtr>&>(const reco::PFTau& tau, bool signal) {
    if (signal){
      return tau.signalPFGammaCands();
    }
    return tau.isolationPFGammaCands();
  }

  template<>
  reco::CandidatePtrVector getGammas<pat::Tau, reco::CandidatePtrVector>(const pat::Tau& tau, bool signal) {
    if(signal){
      return tau.signalGammaCands();
    }
    return tau.isolationGammaCands();
  }

  /// decide if photon candidate is inside the cone to be associated to the tau signal
  bool isInside(float photon_pt, float deta, float dphi) {
    constexpr double stripEtaAssociationDistance_0p95_p0 = 0.197077;
    constexpr double stripEtaAssociationDistance_0p95_p1 = 0.658701;
    constexpr double stripPhiAssociationDistance_0p95_p0 = 0.352476;
    constexpr double stripPhiAssociationDistance_0p95_p1 = 0.707716;
    if(photon_pt==0){
      return false;
    }      if((dphi<0.3  && dphi<std::max(0.05, stripPhiAssociationDistance_0p95_p0*std::pow(photon_pt, -stripPhiAssociationDistance_0p95_p1))) && (deta<0.15 && deta<std::max(0.05, stripEtaAssociationDistance_0p95_p0*std::pow(photon_pt, -stripEtaAssociationDistance_0p95_p1)))){
      return true;
    }
    return false;
  }    
}

namespace reco { namespace tau {
  /// return chi2 of the leading track ==> deprecated? <==
  float lead_track_chi2(const reco::PFTau& tau);
  /// return ratio of energy in ECAL over sum of energy in ECAL and HCAL
  float eratio(const reco::PFTau& tau);
  float eratio(const pat::Tau& tau);
  /// return sum of pt weighted values of distance to tau candidate for all pf photon candidates, 
  /// which are associated to signal; depending on var the distance is in 0=:dr, 1=:deta, 2=:dphi 
  template<class TauType, class PhotonVectorType>
  float pt_weighted_dx(const TauType& tau, int mode = 0, int var = 0, int decaymode = -1) {
    float sum_pt = 0.;
    float sum_dx_pt = 0.;
    float signalrad = std::max(0.05, std::min(0.1, 3./std::max(1., tau.pt())));
    int is3prong = (decaymode==10);
    const auto& cands = getGammas<TauType, PhotonVectorType>(tau, mode < 2);
    for (const auto& cand : cands) {
      // only look at electrons/photons with pT > 0.5
      if (cand->pt() < 0.5){
        continue;
      }
      float dr = reco::deltaR(cand->eta(), cand->phi(), tau.eta(), tau.phi());
      float deta = std::abs(cand->eta() - tau.eta());
      float dphi = std::abs(reco::deltaPhi(cand->phi(), tau.phi()));
      float pt = cand->pt();
      bool flag = isInside(pt, deta, dphi);
      if(is3prong==0){
        if (mode == 2 || (mode == 0 && dr < signalrad) || (mode == 1 && dr > signalrad)) {
          sum_pt += pt;
          if (var == 0)
            sum_dx_pt += pt * dr;
          else if (var == 1)
            sum_dx_pt += pt * deta;
          else if (var == 2)
            sum_dx_pt += pt * dphi;
        }
      }
      else if(is3prong==1){
        if( (mode==2 && flag==false) || (mode==1 && flag==true) || mode==0){
          sum_pt += pt;
          if (var == 0)
            sum_dx_pt += pt * dr;
          else if (var == 1)
            sum_dx_pt += pt * deta;
          else if (var == 2)
            sum_dx_pt += pt * dphi;
        }
      }
    }
    if (sum_pt > 0.){
      return sum_dx_pt/sum_pt;  
    }
    return 0.;
  }
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_dr_signal(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx<reco::PFTau, const std::vector<reco::PFCandidatePtr>&>(tau, 0, 0, dm);
  }
  inline float pt_weighted_dr_signal(const pat::Tau& tau, int dm) {
    return pt_weighted_dx<pat::Tau, reco::CandidatePtrVector>(tau, 0, 0, dm);
  }
  /// return sum of pt weighted values of deta relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_deta_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx<reco::PFTau, const std::vector<reco::PFCandidatePtr>&>(tau, dm==10 ? 2 : 1, 1, dm);
  }
  inline float pt_weighted_deta_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx<pat::Tau, reco::CandidatePtrVector>(tau, dm==10 ? 2 : 1, 1, dm);
  }
  /// return sum of pt weighted values of dphi relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  inline float pt_weighted_dphi_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx<reco::PFTau, const std::vector<reco::PFCandidatePtr>&>(tau, dm==10 ? 2 : 1, 2, dm);
  }
  inline float pt_weighted_dphi_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx<pat::Tau, reco::CandidatePtrVector>(tau, dm==10 ? 2 : 1, 2, dm);
  }  
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are inside an isolation conde but not associated to signal
  inline float pt_weighted_dr_iso(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx<reco::PFTau, const std::vector<reco::PFCandidatePtr>&>(tau, 2, 0, dm);
  }
  inline float pt_weighted_dr_iso(const pat::Tau& tau, int dm) {
    return pt_weighted_dx<pat::Tau, reco::CandidatePtrVector>(tau, 2, 0, dm);
  }
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are inside an isolation conde but not associated to signal
  float pt_weighted_dr_iso(const reco::PFTau& tau, int dm);
  float pt_weighted_dr_iso(const pat::Tau& tau, int dm);
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
