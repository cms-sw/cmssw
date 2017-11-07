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
#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

class TauIdMVAAuxiliaries {
  public:
    /// default constructor
    TauIdMVAAuxiliaries() {};
    /// default destructor
    ~TauIdMVAAuxiliaries() {};
    /// return chi2 of the leading track ==> deprecated? <==
    float tau_leadTrackChi2(const reco::PFTau& tau) const {
      float LeadingTracknormalizedChi2 = 0;
      const reco::PFCandidatePtr& leadingPFCharged = tau.leadPFChargedHadrCand() ;
      if (leadingPFCharged.isNonnull()) {
        reco::TrackRef tref = leadingPFCharged -> trackRef();
        if (tref.isNonnull()) {
          LeadingTracknormalizedChi2 = (float)(tref -> normalizedChi2());
        }
      }
      return LeadingTracknormalizedChi2;
    }

    float tau_leadTrackChi2(const reco::PFBaseTau& tau) const {
      float LeadingTracknormalizedChi2 = 0;
      const auto& leadingPFCharged = edm::Ptr<pat::PackedCandidate>(tau.leadPFChargedHadrCand());
      if (leadingPFCharged.isNonnull() && leadingPFCharged->hasTrackDetails()) {
        const auto& tref = leadingPFCharged->pseudoTrack();
        LeadingTracknormalizedChi2 = (float)(tref.normalizedChi2());
      }
      return LeadingTracknormalizedChi2;
    }

    /// return ratio of energy in ECAL over sum of energy in ECAL and HCAL
    float tau_Eratio(const reco::PFTau& tau) const {
      std::vector<reco::PFCandidatePtr> constsignal = tau.signalPFCands();
      float EcalEnInSignalPFCands = 0;
      float HcalEnInSignalPFCands = 0;
      typedef std::vector <reco::PFCandidatePtr>::iterator constituents_iterator;
      for(constituents_iterator it=constsignal.begin(); it != constsignal.end(); ++it) {
        reco::PFCandidatePtr & icand = *it;
        EcalEnInSignalPFCands += icand -> ecalEnergy();
        HcalEnInSignalPFCands += icand -> hcalEnergy();
      }
      float total = EcalEnInSignalPFCands + HcalEnInSignalPFCands;
      if(total==0){ 
        return -1;
      }
      return EcalEnInSignalPFCands/total;
    }
    float tau_Eratio(const pat::Tau& tau) const {
      float EcalEnInSignalCands = tau.ecalEnergy();
      float HcalEnInSignalCands = tau.hcalEnergy();
      float total = EcalEnInSignalCands + HcalEnInSignalCands;
      if(total == 0){ 
        return -1;
      }
      return EcalEnInSignalCands/total;
    }
    /// return sum of pt weighted values of distance to tau candidate for all pf photon candidates, 
    /// which are associated to signal; depending on var the distance is in 0=:dr, 1=:deta, 2=:dphi 
    float pt_weighted_dx(const reco::PFTau& tau, int mode = 0, int var = 0, int decaymode = -1) const {  
      float sum_pt = 0.;
      float sum_dx_pt = 0.;
      float signalrad = std::max(0.05, std::min(0.1, 3./std::max(1., tau.pt())));
      int is3prong = (decaymode==10);
      const auto& cands = getPFGammas(tau, mode < 2);
      for (const auto& cand : cands) {
        // only look at electrons/photons with pT > 0.5
        if ((float)cand->pt() < 0.5){
          continue;
        }
        float dr = reco::deltaR((float)cand->eta(),(float)cand->phi(),(float)tau.eta(),(float)tau.phi());
        float deta = std::abs((float)cand->eta() - (float)tau.eta());
        float dphi = std::abs(reco::deltaPhi((float)cand->phi(), (float)tau.phi()));
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
    float pt_weighted_dx(const pat::Tau& tau, int mode = 0, int var = 0, int decaymode = -1) const {
      float sum_pt = 0.;
      float sum_dx_pt = 0.;
      float signalrad = std::max(0.05, std::min(0.1, 3./std::max(1., tau.pt())));
      int is3prong = (decaymode==10);
      const auto cands = getGammas(tau, mode < 2);
      for (const auto& cand : cands) {
        // only look at electrons/photons with pT > 0.5
        if (cand->pt() < 0.5){
          continue;
        }
        float dr = reco::deltaR(*cand, tau);
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
    float tau_pt_weighted_dr_signal(const reco::PFTau& tau, int dm) const {
      return pt_weighted_dx(tau, 0, 0, dm);
    }
    float tau_pt_weighted_dr_signal(const pat::Tau& tau, int dm) const {
      return pt_weighted_dx(tau, 0, 0, dm);
    }
    /// return sum of pt weighted values of deta relative to tau candidate for all pf photon candidates,
    /// which are associated to signal
    float tau_pt_weighted_deta_strip(const reco::PFTau& tau, int dm) const {
      return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
    }
    float tau_pt_weighted_deta_strip(const pat::Tau& tau, int dm) const {
      return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
    }
    /// return sum of pt weighted values of dphi relative to tau candidate for all pf photon candidates,
    /// which are associated to signal
    float tau_pt_weighted_dphi_strip(const reco::PFTau& tau, int dm) const {
      return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
    }
    float tau_pt_weighted_dphi_strip(const pat::Tau& tau, int dm) const {
      return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
    }  
    /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
    /// which are inside an isolation conde but not associated to signal
    float tau_pt_weighted_dr_iso(const reco::PFTau& tau, int dm) const {
      return pt_weighted_dx(tau, 2, 0, dm);
    }
    float tau_pt_weighted_dr_iso(const pat::Tau& tau, int dm) const {
      return pt_weighted_dx(tau, 2, 0, dm);
    }
    /// return total number of pf photon candidates with pT>500 MeV, which are associated to signal
    unsigned int tau_n_photons_total(const reco::PFTau& tau) const {
      unsigned int n_photons = 0;
      for (auto& cand : tau.signalPFGammaCands()) {
        if ((float)cand->pt() > 0.5)
          ++n_photons;
      }
      for (auto& cand : tau.isolationPFGammaCands()) {
        if ((float)cand->pt() > 0.5)
          ++n_photons;
      }
      return n_photons;
    }
    unsigned int tau_n_photons_total(const pat::Tau& tau) const {
      unsigned int n_photons = 0;
      for (auto& cand : tau.signalGammaCands()) {
        if (cand->pt() > 0.5) 
          ++n_photons;
      }
      for (auto& cand : tau.isolationGammaCands()) {  
        if (cand->pt() > 0.5)
	  ++n_photons;
      }
      return n_photons;
    }
  
  private:
    /// return pf photon candidates that are associated to signal
    const std::vector<reco::PFCandidatePtr>& getPFGammas(const reco::PFTau& tau, bool signal = true) const {
      if (signal){
        return tau.signalPFGammaCands();
      }
      return tau.isolationPFGammaCands();
    }
    reco::CandidatePtrVector getGammas(const pat::Tau& tau, bool signal = true) const {
      if(signal){
        return tau.signalGammaCands();
      }
      return tau.isolationGammaCands();
    }
    /// decide if photon candidate is inside the cone to be associated to the tau signal
    bool isInside(float photon_pt, float deta, float dphi) const {
      const double stripEtaAssociationDistance_0p95_p0 = 0.197077;
      const double stripEtaAssociationDistance_0p95_p1 = 0.658701;
      const double stripPhiAssociationDistance_0p95_p0 = 0.352476;
      const double stripPhiAssociationDistance_0p95_p1 = 0.707716;
      if(photon_pt==0){
        return false;
      }
      if((dphi<0.3  && dphi<std::max(0.05, stripPhiAssociationDistance_0p95_p0*std::pow(photon_pt, -stripPhiAssociationDistance_0p95_p1))) && \
         (deta<0.15 && deta<std::max(0.05, stripEtaAssociationDistance_0p95_p0*std::pow(photon_pt, -stripEtaAssociationDistance_0p95_p1)))){
        return true;
      }
      return false;
    }      
};
