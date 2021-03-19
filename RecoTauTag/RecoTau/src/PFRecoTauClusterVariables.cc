#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"

namespace {
  struct PFTau_traits {
    typedef reco::PFTau Tau_t;
    typedef const std::vector<reco::CandidatePtr>& Ret_t;
  };
  struct PATTau_traits {
    typedef pat::Tau Tau_t;
    typedef reco::CandidatePtrVector Ret_t;
  };

  template <typename T>
  typename T::Ret_t getGammas_T(const typename T::Tau_t& tau, bool signal) {
    return typename T::Ret_t();
  }
  /// return pf photon candidates that are associated to signal
  template <>
  const std::vector<reco::CandidatePtr>& getGammas_T<PFTau_traits>(const reco::PFTau& tau, bool signal) {
    if (signal) {
      return tau.signalGammaCands();
    }
    return tau.isolationGammaCands();
  }

  template <>
  reco::CandidatePtrVector getGammas_T<PATTau_traits>(const pat::Tau& tau, bool signal) {
    if (signal) {
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
    if (photon_pt == 0) {
      return false;
    }
    if ((dphi < 0.3 && dphi < std::max(0.05,
                                       stripPhiAssociationDistance_0p95_p0 *
                                           std::pow(photon_pt, -stripPhiAssociationDistance_0p95_p1))) &&
        (deta < 0.15 && deta < std::max(0.05,
                                        stripEtaAssociationDistance_0p95_p0 *
                                            std::pow(photon_pt, -stripEtaAssociationDistance_0p95_p1)))) {
      return true;
    }
    return false;
  }
}  // namespace

namespace reco {
  namespace tau {
    /// return chi2 of the leading track ==> deprecated? <==
    float lead_track_chi2(const reco::PFTau& tau) {
      float LeadingTracknormalizedChi2 = 0;
      const reco::CandidatePtr& leadingCharged = tau.leadChargedHadrCand();
      if (leadingCharged.isNonnull()) {
        const reco::PFCandidate* pfcand = dynamic_cast<const reco::PFCandidate*>(leadingCharged.get());
        if (pfcand != nullptr) {
          reco::TrackRef tref = pfcand->trackRef();
          if (tref.isNonnull()) {
            LeadingTracknormalizedChi2 = tref->normalizedChi2();
          }
        } else {
          const pat::PackedCandidate* patcand = dynamic_cast<const pat::PackedCandidate*>(leadingCharged.get());
          if (patcand != nullptr && patcand->hasTrackDetails()) {
            LeadingTracknormalizedChi2 = patcand->pseudoTrack().normalizedChi2();
          }
        }
      }
      return LeadingTracknormalizedChi2;
    }
    /// return ratio of energy in ECAL over sum of energy in ECAL and HCAL
    float eratio(const reco::PFTau& tau) {
      float ecal_en_in_signal_pf_cands = 0;
      float hcal_en_in_signal_pf_cands = 0;
      for (const auto& signal_cand : tau.signalCands()) {
        const reco::PFCandidate* signal_pfcand = dynamic_cast<const reco::PFCandidate*>(signal_cand.get());
        if (signal_pfcand != nullptr) {
          ecal_en_in_signal_pf_cands += signal_pfcand->ecalEnergy();
          hcal_en_in_signal_pf_cands += signal_pfcand->hcalEnergy();
        } else {
          // TauReco@MiniAOD: individual ECAL and HCAL energies recovered from fractions
          const pat::PackedCandidate* signal_pcand = dynamic_cast<const pat::PackedCandidate*>(signal_cand.get());
          assert(signal_pcand);  // Taus are built either from reco::PFCandidates or pat::PackedCandidates
          float calo_en = signal_pcand->caloFraction() * signal_pcand->energy();
          ecal_en_in_signal_pf_cands += calo_en * (1. - signal_pcand->hcalFraction());
          hcal_en_in_signal_pf_cands += calo_en * signal_pcand->hcalFraction();
        }
      }
      float total = ecal_en_in_signal_pf_cands + hcal_en_in_signal_pf_cands;
      if (total == 0.) {
        return -1.;
      }
      return ecal_en_in_signal_pf_cands / total;
    }
    float eratio(const pat::Tau& tau) {
      float ecal_en_in_signal_cands = tau.ecalEnergy();
      float hcal_en_in_signal_cands = tau.hcalEnergy();
      float total = ecal_en_in_signal_cands + hcal_en_in_signal_cands;
      if (total == 0.) {
        return -1.;
      }
      return ecal_en_in_signal_cands / total;
    }
    /// return sum of pt weighted values of distance to tau candidate for all pf photon candidates,
    /// which are associated to signal; depending on var the distance is in 0=:dr, 1=:deta, 2=:dphi
    template <typename T>
    float pt_weighted_dx_T(const typename T::Tau_t& tau, int mode, int var, int decaymode) {
      float sum_pt = 0.;
      float sum_dx_pt = 0.;
      float signalrad = std::max(0.05, std::min(0.1, 3. / std::max(1., tau.pt())));
      int is3prong = (decaymode == 10);
      const auto& cands = getGammas_T<T>(tau, mode < 2);
      for (const auto& cand : cands) {
        // only look at electrons/photons with pT > 0.5
        if (cand->pt() < 0.5) {
          continue;
        }
        float dr = reco::deltaR(cand->eta(), cand->phi(), tau.eta(), tau.phi());
        float deta = std::abs(cand->eta() - tau.eta());
        float dphi = std::abs(reco::deltaPhi(cand->phi(), tau.phi()));
        float pt = cand->pt();
        bool flag = isInside(pt, deta, dphi);
        if (is3prong == 0) {
          if (mode == 2 || (mode == 0 && dr < signalrad) || (mode == 1 && dr > signalrad)) {
            sum_pt += pt;
            if (var == 0)
              sum_dx_pt += pt * dr;
            else if (var == 1)
              sum_dx_pt += pt * deta;
            else if (var == 2)
              sum_dx_pt += pt * dphi;
          }
        } else if (is3prong == 1) {
          if ((mode == 2 && flag == false) || (mode == 1 && flag == true) || mode == 0) {
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
      if (sum_pt > 0.) {
        return sum_dx_pt / sum_pt;
      }
      return 0.;
    }
    float pt_weighted_dx(const reco::PFTau& tau, int mode, int var, int decaymode) {
      return pt_weighted_dx_T<PFTau_traits>(tau, mode, var, decaymode);
    }
    float pt_weighted_dx(const pat::Tau& tau, int mode, int var, int decaymode) {
      return pt_weighted_dx_T<PATTau_traits>(tau, mode, var, decaymode);
    }
    /// return total number of pf photon candidates with pT>500 MeV, which are associated to signal
    unsigned int n_photons_total(const reco::PFTau& tau) {
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
    unsigned int n_photons_total(const pat::Tau& tau) {
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

    bool fillIsoMVARun2Inputs(float* mvaInput,
                              const pat::Tau& tau,
                              int mvaOpt,
                              const std::string& nameCharged,
                              const std::string& nameNeutral,
                              const std::string& namePu,
                              const std::string& nameOutside,
                              const std::string& nameFootprint) {
      int tauDecayMode = tau.decayMode();
      const float mTau = 1.77682;

      if (((mvaOpt == kOldDMwoLT || mvaOpt == kOldDMwLT || mvaOpt == kDBoldDMwLT || mvaOpt == kPWoldDMwLT ||
            mvaOpt == kDBoldDMwLTwGJ) &&
           (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 10)) ||
          ((mvaOpt == kDBnewDMwLTwGJPhase2 || mvaOpt == kNewDMwoLT || mvaOpt == kNewDMwLT || mvaOpt == kDBnewDMwLT ||
            mvaOpt == kPWnewDMwLT || mvaOpt == kDBnewDMwLTwGJ) &&
           (tauDecayMode == 0 || tauDecayMode == 1 || tauDecayMode == 2 || tauDecayMode == 5 || tauDecayMode == 6 ||
            tauDecayMode == 10 || tauDecayMode == 11))) {
        float chargedIsoPtSum = tau.tauID(nameCharged);
        float neutralIsoPtSum = tau.tauID(nameNeutral);
        float puCorrPtSum = tau.tauID(namePu);
        float photonPtSumOutsideSignalCone = tau.tauID(nameOutside);
        float footprintCorrection = tau.tauID(nameFootprint);

        float decayDistX = tau.flightLength().x();
        float decayDistY = tau.flightLength().y();
        float decayDistZ = tau.flightLength().z();
        float decayDistMag = std::sqrt(decayDistX * decayDistX + decayDistY * decayDistY + decayDistZ * decayDistZ);

        // --- The following 5 variables differ slightly between AOD & MiniAOD
        //     because they are recomputed using packedCandidates saved in the tau
        float nPhoton = reco::tau::n_photons_total(tau);
        float ptWeightedDetaStrip = reco::tau::pt_weighted_deta_strip(tau, tauDecayMode);
        float ptWeightedDphiStrip = reco::tau::pt_weighted_dphi_strip(tau, tauDecayMode);
        float ptWeightedDrSignal = reco::tau::pt_weighted_dr_signal(tau, tauDecayMode);
        float ptWeightedDrIsolation = reco::tau::pt_weighted_dr_iso(tau, tauDecayMode);
        // ---
        float leadingTrackChi2 = tau.leadingTrackNormChi2();
        float eRatio = reco::tau::eratio(tau);

        // Difference between measured and maximally allowed Gottfried-Jackson angle
        float gjAngleDiff = -999;
        if (tauDecayMode == 10) {
          double mAOne = tau.p4().M();
          double pAOneMag = tau.p();
          double argumentThetaGJmax = (std::pow(mTau, 2) - std::pow(mAOne, 2)) / (2 * mTau * pAOneMag);
          double argumentThetaGJmeasured =
              (tau.p4().px() * decayDistX + tau.p4().py() * decayDistY + tau.p4().pz() * decayDistZ) /
              (pAOneMag * decayDistMag);
          if (std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1.) {
            double thetaGJmax = std::asin(argumentThetaGJmax);
            double thetaGJmeasured = std::acos(argumentThetaGJmeasured);
            gjAngleDiff = thetaGJmeasured - thetaGJmax;
          }
        }

        if (mvaOpt == kDBnewDMwLTwGJPhase2) {
          mvaInput[0] = tau.pt();
          mvaInput[1] = std::abs(tau.eta());
          mvaInput[2] = chargedIsoPtSum;               //tauID("chargedIsoPtSum");
          mvaInput[3] = neutralIsoPtSum;               //tauID("neutralIsoPtSum");
          mvaInput[4] = puCorrPtSum;                   //tauID("puCorrPtSum");
          mvaInput[5] = photonPtSumOutsideSignalCone;  //tauID("photonPtSumOutsideSignalCone");
          mvaInput[6] = tauDecayMode;                  //tau.decayMode();
          mvaInput[7] = tau.signalGammaCands().size();
          mvaInput[8] = tau.isolationGammaCands().size();

          float sigCands_pt = 0.;
          float sigCands_dr, sigCands_deta, sigCands_dphi;
          sigCands_dr = sigCands_deta = sigCands_dphi = 0.;
          for (const auto& j : tau.signalGammaCands()) {
            const float dr = reco::deltaR(tau, *j);
            const float deta = std::abs(tau.eta() - j->eta());
            const float dphi = std::abs(reco::deltaPhi(tau.phi(), j->phi()));
            const float pt_ = j->pt();
            sigCands_dr += dr * pt_;
            sigCands_deta += deta * pt_;
            sigCands_dphi += dphi * pt_;
            sigCands_pt += pt_;
          }
          if (sigCands_pt > 0.) {
            sigCands_dr = sigCands_dr / sigCands_pt;
            sigCands_deta = sigCands_deta / sigCands_pt;
            sigCands_dphi = sigCands_dphi / sigCands_pt;
          } else {
            sigCands_dr = sigCands_deta = sigCands_dphi = -0.1;
          }
          float isoCands_pt = 0.;
          float isoCands_dr, isoCands_deta, isoCands_dphi;
          isoCands_dr = isoCands_deta = isoCands_dphi = 0.;
          for (const auto& j : tau.isolationGammaCands()) {
            const float dr = reco::deltaR(tau, *j);
            const float deta = std::abs(tau.eta() - j->eta());
            const float dphi = std::abs(reco::deltaPhi(tau.phi(), j->phi()));
            const float pt_ = j->pt();
            isoCands_dr += dr * pt_;
            isoCands_deta += deta * pt_;
            isoCands_dphi += dphi * pt_;
            isoCands_pt += pt_;
          }
          if (isoCands_pt > 0.) {
            isoCands_dr = isoCands_dr / isoCands_pt;
            isoCands_deta = isoCands_deta / isoCands_pt;
            isoCands_dphi = isoCands_dphi / isoCands_pt;
          } else {
            isoCands_dr = isoCands_deta = isoCands_dphi = -0.1;
          }
          mvaInput[9] = isoCands_deta;
          mvaInput[10] = isoCands_dphi;
          mvaInput[11] = isoCands_dr;
          mvaInput[12] = sigCands_deta;
          mvaInput[13] = sigCands_dphi;
          mvaInput[14] = sigCands_dr;

          float e = tau.hcalEnergy() + tau.ecalEnergy();
          e > 0. ? e = tau.ecalEnergy() / e : e = -1.;
          mvaInput[15] = e;
          mvaInput[16] = tau.dxy() >= 0. ? +1 : -1;
          mvaInput[17] = sqrt(std::abs(tau.dxy()));
          mvaInput[18] = std::abs(tau.dxy_Sig());
          mvaInput[19] = tau.ip3d() >= 0. ? +1 : -1;
          mvaInput[20] = sqrt(std::abs(tau.ip3d()));
          mvaInput[21] = std::abs(tau.ip3d_Sig());
          mvaInput[22] = (tau.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput[23] = decayDistMag;  //sqrt(tau.flightLength().Mag2());
          mvaInput[24] = tau.flightLengthSig();
          mvaInput[25] = leadingTrackChi2;  //tau.leadingTrackNormChi2();

          float thetaGJmax, thetaGJ;
          if (decayDistMag > 0. && tau.hasSecondaryVertex()) {
            const float mAOne = tau.p4().M();
            const float pAOneMag = tau.p();
            thetaGJmax = (mTau * mTau - mAOne * mAOne) / (2. * mTau * pAOneMag);
            thetaGJmax = asin(thetaGJmax);
            thetaGJ = (tau.px() * tau.flightLength().x() + tau.py() * tau.flightLength().y() +
                       tau.pz() * tau.flightLength().z()) /
                      (pAOneMag * decayDistMag);
            thetaGJ = acos(thetaGJ);
            if (std::isnan(thetaGJ))
              thetaGJ = -16.;
            if (std::isnan(thetaGJmax))
              thetaGJmax = -11.;
          } else {
            thetaGJ = -15.;
            thetaGJmax = -10.;
          }
          mvaInput[26] = thetaGJ - thetaGJmax;

          mvaInput[27] = 0;
          mvaInput[28] = 10.;
          mvaInput[29] = 10.;
          if (tau.leadChargedHadrCand().isNonnull()) {
            if (tau.leadChargedHadrCand()->bestTrack()) {
              const float trackdxy = tau.leadChargedHadrCand()->bestTrack()->dxy();
              const float trackdxy_err = tau.leadChargedHadrCand()->bestTrack()->dxyError();
              mvaInput[27] = trackdxy >= 0. ? +1 : -1;
              mvaInput[28] = sqrt(std::abs(trackdxy));
              mvaInput[29] = std::abs(trackdxy / trackdxy_err);
            }
          }
        }
        if (mvaOpt == kOldDMwoLT || mvaOpt == kNewDMwoLT) {
          mvaInput[0] = std::log(std::max(1.f, (float)tau.pt()));
          mvaInput[1] = std::abs((float)tau.eta());
          mvaInput[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput[3] = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f * puCorrPtSum));
          mvaInput[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput[5] = tauDecayMode;
        } else if (mvaOpt == kOldDMwLT || mvaOpt == kNewDMwLT) {
          mvaInput[0] = std::log(std::max(1.f, (float)tau.pt()));
          mvaInput[1] = std::abs((float)tau.eta());
          mvaInput[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput[3] = std::log(std::max(1.e-2f, neutralIsoPtSum - 0.125f * puCorrPtSum));
          mvaInput[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput[5] = tauDecayMode;
          mvaInput[6] = std::copysign(+1.f, tau.dxy());
          mvaInput[7] = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
          mvaInput[8] = std::min(10.f, std::abs(tau.dxy_Sig()));
          mvaInput[9] = (tau.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput[10] = std::sqrt(decayDistMag);
          mvaInput[11] = std::min(10.f, tau.flightLengthSig());
        } else if (mvaOpt == kDBoldDMwLT || mvaOpt == kDBnewDMwLT) {
          mvaInput[0] = std::log(std::max(1.f, (float)tau.pt()));
          mvaInput[1] = std::abs((float)tau.eta());
          mvaInput[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput[6] = tauDecayMode;
          mvaInput[7] = std::min(30.f, nPhoton);
          mvaInput[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput[12] = std::min(100.f, leadingTrackChi2);
          mvaInput[13] = std::min(1.f, eRatio);
          mvaInput[14] = std::copysign(+1.f, tau.dxy());
          mvaInput[15] = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
          mvaInput[16] = std::min(10.f, std::abs(tau.dxy_Sig()));
          mvaInput[17] = std::copysign(+1.f, tau.ip3d());
          mvaInput[18] = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
          mvaInput[19] = std::min(10.f, std::abs(tau.ip3d_Sig()));
          mvaInput[20] = (tau.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput[21] = std::sqrt(decayDistMag);
          mvaInput[22] = std::min(10.f, tau.flightLengthSig());
        } else if (mvaOpt == kPWoldDMwLT || mvaOpt == kPWnewDMwLT) {
          mvaInput[0] = std::log(std::max(1.f, (float)tau.pt()));
          mvaInput[1] = std::abs((float)tau.eta());
          mvaInput[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput[4] = std::log(std::max(1.e-2f, footprintCorrection));
          mvaInput[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput[6] = tauDecayMode;
          mvaInput[7] = std::min(30.f, nPhoton);
          mvaInput[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput[12] = std::min(100.f, leadingTrackChi2);
          mvaInput[13] = std::min(1.f, eRatio);
          mvaInput[14] = std::copysign(+1.f, tau.dxy());
          mvaInput[15] = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
          mvaInput[16] = std::min(10.f, std::abs(tau.dxy_Sig()));
          mvaInput[17] = std::copysign(+1.f, tau.ip3d());
          mvaInput[18] = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
          mvaInput[19] = std::min(10.f, std::abs(tau.ip3d_Sig()));
          mvaInput[20] = (tau.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput[21] = std::sqrt(decayDistMag);
          mvaInput[22] = std::min(10.f, tau.flightLengthSig());
        } else if (mvaOpt == kDBoldDMwLTwGJ || mvaOpt == kDBnewDMwLTwGJ) {
          mvaInput[0] = std::log(std::max(1.f, (float)tau.pt()));
          mvaInput[1] = std::abs((float)tau.eta());
          mvaInput[2] = std::log(std::max(1.e-2f, chargedIsoPtSum));
          mvaInput[3] = std::log(std::max(1.e-2f, neutralIsoPtSum));
          mvaInput[4] = std::log(std::max(1.e-2f, puCorrPtSum));
          mvaInput[5] = std::log(std::max(1.e-2f, photonPtSumOutsideSignalCone));
          mvaInput[6] = tauDecayMode;
          mvaInput[7] = std::min(30.f, nPhoton);
          mvaInput[8] = std::min(0.5f, ptWeightedDetaStrip);
          mvaInput[9] = std::min(0.5f, ptWeightedDphiStrip);
          mvaInput[10] = std::min(0.5f, ptWeightedDrSignal);
          mvaInput[11] = std::min(0.5f, ptWeightedDrIsolation);
          mvaInput[12] = std::min(1.f, eRatio);
          mvaInput[13] = std::copysign(+1.f, tau.dxy());
          mvaInput[14] = std::sqrt(std::min(1.f, std::abs(tau.dxy())));
          mvaInput[15] = std::min(10.f, std::abs(tau.dxy_Sig()));
          mvaInput[16] = std::copysign(+1.f, tau.ip3d());
          mvaInput[17] = std::sqrt(std::min(1.f, std::abs(tau.ip3d())));
          mvaInput[18] = std::min(10.f, std::abs(tau.ip3d_Sig()));
          mvaInput[19] = (tau.hasSecondaryVertex()) ? 1. : 0.;
          mvaInput[20] = std::sqrt(decayDistMag);
          mvaInput[21] = std::min(10.f, tau.flightLengthSig());
          mvaInput[22] = std::max(-1.f, gjAngleDiff);
        }

        return true;
      }
      return false;
    }

  }  // namespace tau
}  // namespace reco
