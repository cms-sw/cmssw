#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

namespace {

  /// return pf photon candidates that are associated to signal
  const std::vector<reco::PFCandidatePtr>& getPFGammas(const reco::PFTau& tau, bool signal) {
    if (signal){
      return tau.signalPFGammaCands();
    }
    return tau.isolationPFGammaCands();
  }
  reco::CandidatePtrVector getGammas(const pat::Tau& tau, bool signal) {
    if(signal){
      return tau.signalGammaCands();
    }
    return tau.isolationGammaCands();
  }
  /// decide if photon candidate is inside the cone to be associated to the tau signal
  bool isInside(float photon_pt, float deta, float dphi) {
    const double stripEtaAssociationDistance_0p95_p0 = 0.197077;
    const double stripEtaAssociationDistance_0p95_p1 = 0.658701;
    const double stripPhiAssociationDistance_0p95_p0 = 0.352476;
    const double stripPhiAssociationDistance_0p95_p1 = 0.707716;
    if(photon_pt==0){
      return false;
    }      if((dphi<0.3  && dphi<std::max(0.05, stripPhiAssociationDistance_0p95_p0*std::pow(photon_pt, -stripPhiAssociationDistance_0p95_p1))) && (deta<0.15 && deta<std::max(0.05, stripEtaAssociationDistance_0p95_p0*std::pow(photon_pt, -stripEtaAssociationDistance_0p95_p1)))){
      return true;
    }
    return false;
  }    
}

namespace reco { namespace tau { namespace mva {
  /// return chi2 of the leading track ==> deprecated? <==
  float tau_leadTrackChi2(const reco::PFTau& tau) {
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
  /// return ratio of energy in ECAL over sum of energy in ECAL and HCAL
  float tau_Eratio(const reco::PFTau& tau) {
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
  float tau_Eratio(const pat::Tau& tau) {
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
  float pt_weighted_dx(const reco::PFTau& tau, int mode, int var, int decaymode) {  
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
  float pt_weighted_dx(const pat::Tau& tau, int mode, int var, int decaymode) {
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
  float tau_pt_weighted_dr_signal(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, 0, 0, dm);
  }
  float tau_pt_weighted_dr_signal(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, 0, 0, dm);
  }
  /// return sum of pt weighted values of deta relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  float tau_pt_weighted_deta_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
  }
  float tau_pt_weighted_deta_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 1, dm);
  }
  /// return sum of pt weighted values of dphi relative to tau candidate for all pf photon candidates,
  /// which are associated to signal
  float tau_pt_weighted_dphi_strip(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
  }
  float tau_pt_weighted_dphi_strip(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, dm==10 ? 2 : 1, 2, dm);
  }  
  /// return sum of pt weighted values of dr relative to tau candidate for all pf photon candidates,
  /// which are inside an isolation conde but not associated to signal
  float tau_pt_weighted_dr_iso(const reco::PFTau& tau, int dm) {
    return pt_weighted_dx(tau, 2, 0, dm);
  }
  float tau_pt_weighted_dr_iso(const pat::Tau& tau, int dm) {
    return pt_weighted_dx(tau, 2, 0, dm);
  }
  /// return total number of pf photon candidates with pT>500 MeV, which are associated to signal
  unsigned int tau_n_photons_total(const reco::PFTau& tau) {
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
  unsigned int tau_n_photons_total(const pat::Tau& tau) {
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

  bool fillMVAInputs(float* mvaInput, const pat::Tau& tau, int mvaOpt, const std::string nameCharged, const std::string   nameNeutral, const std::string namePu, const std::string nameOutside, const std::string nameFootprint) 
  {
    int tauDecayMode = tau.decayMode();
    
  if ( ((mvaOpt == kOldDMwoLT || mvaOpt == kOldDMwLT || mvaOpt == kDBoldDMwLT || mvaOpt == kPWoldDMwLT || mvaOpt ==   kDBoldDMwLTwGJ)
          && (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 10))
         ||
       ((mvaOpt == kNewDMwoLT || mvaOpt == kNewDMwLT || mvaOpt == kDBnewDMwLT || mvaOpt == kPWnewDMwLT || mvaOpt ==   kDBnewDMwLTwGJ)
        && (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 5 || tauDecayMode == 6 ||   tauDecayMode == 10 || tauDecayMode == 11))
    ) {
  	
      float chargedIsoPtSum = tau.tauID(nameCharged);
      float neutralIsoPtSum = tau.tauID(nameNeutral);
      float puCorrPtSum     = tau.tauID(namePu);
      float photonPtSumOutsideSignalCone = tau.tauID(nameOutside);
      float footprintCorrection = tau.tauID(nameFootprint);
  		
      float decayDistX = tau.flightLength().x();
      float decayDistY = tau.flightLength().y();
      float decayDistZ = tau.flightLength().z();
      float decayDistMag = std::sqrt(decayDistX*decayDistX + decayDistY*decayDistY + decayDistZ*decayDistZ);
  		
      // --- The following 5 variables differ slightly between AOD & MiniAOD
      //     because they are recomputed using packedCandidates saved in the tau
      float nPhoton = (float)reco::tau::mva::tau_n_photons_total(tau);
      float ptWeightedDetaStrip = reco::tau::mva::tau_pt_weighted_deta_strip(tau, tauDecayMode);
      float ptWeightedDphiStrip = reco::tau::mva::tau_pt_weighted_dphi_strip(tau, tauDecayMode);
      float ptWeightedDrSignal = reco::tau::mva::tau_pt_weighted_dr_signal(tau, tauDecayMode);
      float ptWeightedDrIsolation = reco::tau::mva::tau_pt_weighted_dr_iso(tau, tauDecayMode);
      // ---
      float leadingTrackChi2 = tau.leadingTrackNormChi2();
      float eRatio = reco::tau::mva::tau_Eratio(tau);
  
      // Difference between measured and maximally allowed Gottfried-Jackson angle
      float gjAngleDiff = -999;
      if ( tauDecayMode == 10 ) {
          double mTau = 1.77682;
          double mAOne = tau.p4().M();
          double pAOneMag = tau.p();
          double argumentThetaGJmax = (std::pow(mTau,2) - std::pow(mAOne,2) ) / ( 2 * mTau * pAOneMag );
        double argumentThetaGJmeasured = ( tau.p4().px() * decayDistX + tau.p4().py() * decayDistY + tau.p4().pz() * decayDistZ   ) / ( pAOneMag * decayDistMag );
          if ( std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1. ) {
              double thetaGJmax = std::asin( argumentThetaGJmax );
              double thetaGJmeasured = std::acos( argumentThetaGJmeasured );
              gjAngleDiff = thetaGJmeasured - thetaGJmax;
          }
      }
  		
      if ( mvaOpt == kOldDMwoLT || mvaOpt == kNewDMwoLT ) {
        mvaInput[0]  = std::log(std::max(1.f, (float)tau.pt()));
        mvaInput[1]  = std::abs((float)tau.eta());
        mvaInput[2]  = std::log(std::max(1.e-2f, chargedIsoPtSum));
        mvaInput[3]  = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f*puCorrPtSum));
        mvaInput[4]  = std::log(std::max(1.e-2f, puCorrPtSum));
        mvaInput[5]  = tauDecayMode;
      } else if ( mvaOpt == kOldDMwLT || mvaOpt == kNewDMwLT  ) {
        mvaInput[0]  = std::log(std::max(1.f, (float)tau.pt()));
        mvaInput[1]  = std::abs((float)tau.eta());
        mvaInput[2]  = std::log(std::max(1.e-2f, chargedIsoPtSum));
        mvaInput[3]  = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f*puCorrPtSum));
        mvaInput[4]  = std::log(std::max(1.e-2f, puCorrPtSum));
        mvaInput[5]  = tauDecayMode;
        mvaInput[6]  = std::copysign(+1.f, tau.dxy());
        mvaInput[7]  = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
        mvaInput[8]  = std::min(10.f, std::abs(tau.dxy_Sig()));
        mvaInput[9]  = ( tau.hasSecondaryVertex() ) ? 1. : 0.;
        mvaInput[10] = std::sqrt(decayDistMag);
        mvaInput[11] = std::min(10.f, tau.flightLengthSig());
      } else if ( mvaOpt == kDBoldDMwLT || mvaOpt == kDBnewDMwLT ) {
        mvaInput[0]  = std::log(std::max(1.f, (float)tau.pt()));
        mvaInput[1]  = std::abs((float)tau.eta());
        mvaInput[2]  = std::log(std::max(1.e-2f, chargedIsoPtSum));
        mvaInput[3]  = std::log(std::max(1.e-2f, neutralIsoPtSum));
        mvaInput[4]  = std::log(std::max(1.e-2f, puCorrPtSum));
        mvaInput[5]  = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
        mvaInput[6]  = tauDecayMode;
        mvaInput[7]  = std::min(30.f, nPhoton);
        mvaInput[8]  = std::min(0.5f, ptWeightedDetaStrip);
        mvaInput[9]  = std::min(0.5f, ptWeightedDphiStrip);
        mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
        mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
        mvaInput[12] = std::min(100.f, leadingTrackChi2);
        mvaInput[13] = std::min(1.f, eRatio);
        mvaInput[14]  = std::copysign(+1.f, tau.dxy());
        mvaInput[15]  = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
        mvaInput[16]  = std::min(10.f, std::abs(tau.dxy_Sig()));
        mvaInput[17]  = std::copysign(+1.f, tau.ip3d());
        mvaInput[18]  = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
        mvaInput[19]  = std::min(10.f, std::abs(tau.ip3d_Sig()));
        mvaInput[20]  = ( tau.hasSecondaryVertex() ) ? 1. : 0.;
        mvaInput[21] = std::sqrt(decayDistMag);
        mvaInput[22] = std::min(10.f, tau.flightLengthSig());
      } else if ( mvaOpt == kPWoldDMwLT || mvaOpt == kPWnewDMwLT ) {
        mvaInput[0]  = std::log(std::max(1.f, (float)tau.pt()));
        mvaInput[1]  = std::abs((float)tau.eta());
        mvaInput[2]  = std::log(std::max(1.e-2f, chargedIsoPtSum));
        mvaInput[3]  = std::log(std::max(1.e-2f, neutralIsoPtSum));
        mvaInput[4]  = std::log(std::max(1.e-2f, footprintCorrection));
        mvaInput[5]  = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
        mvaInput[6]  = tauDecayMode;
        mvaInput[7]  = std::min(30.f, nPhoton);
        mvaInput[8]  = std::min(0.5f, ptWeightedDetaStrip);
        mvaInput[9]  = std::min(0.5f, ptWeightedDphiStrip);
        mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
        mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
        mvaInput[12] = std::min(100.f, leadingTrackChi2);
        mvaInput[13] = std::min(1.f, eRatio);
        mvaInput[14]  = std::copysign(+1.f, tau.dxy());
        mvaInput[15]  = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
        mvaInput[16]  = std::min(10.f, std::abs(tau.dxy_Sig()));
        mvaInput[17]  = std::copysign(+1.f, tau.ip3d());
        mvaInput[18]  = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
        mvaInput[19]  = std::min(10.f, std::abs(tau.ip3d_Sig()));
        mvaInput[20]  = ( tau.hasSecondaryVertex() ) ? 1. : 0.;
        mvaInput[21] = std::sqrt(decayDistMag);
        mvaInput[22] = std::min(10.f, tau.flightLengthSig());
      } else if ( mvaOpt == kDBoldDMwLTwGJ || mvaOpt == kDBnewDMwLTwGJ ) {
        mvaInput[0]  = std::log(std::max(1.f, (float)tau.pt()));
        mvaInput[1]  = std::abs((float)tau.eta());
        mvaInput[2]  = std::log(std::max(1.e-2f, chargedIsoPtSum));
        mvaInput[3]  = std::log(std::max(1.e-2f, neutralIsoPtSum));
        mvaInput[4]  = std::log(std::max(1.e-2f, puCorrPtSum));
        mvaInput[5]  = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
        mvaInput[6]  = tauDecayMode;
        mvaInput[7]  = std::min(30.f, nPhoton);
        mvaInput[8]  = std::min(0.5f, ptWeightedDetaStrip);
        mvaInput[9]  = std::min(0.5f, ptWeightedDphiStrip);
        mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
        mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
        mvaInput[12] = std::min(1.f, eRatio);
        mvaInput[13]  = std::copysign(+1.f, tau.dxy());
        mvaInput[14]  = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
        mvaInput[15]  = std::min(10.f, std::abs(tau.dxy_Sig()));
        mvaInput[16]  = std::copysign(+1.f, tau.ip3d());
        mvaInput[17]  = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
        mvaInput[18]  = std::min(10.f, std::abs(tau.ip3d_Sig()));
        mvaInput[19]  = ( tau.hasSecondaryVertex() ) ? 1. : 0.;
        mvaInput[20] = std::sqrt(decayDistMag);
        mvaInput[21] = std::min(10.f, tau.flightLengthSig());
        mvaInput[22] = std::max(-1.f, gjAngleDiff);
      }
  
      return true;
    }
    return false;
  }

}}} // namespaces