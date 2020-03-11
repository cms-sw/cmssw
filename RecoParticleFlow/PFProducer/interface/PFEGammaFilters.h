#ifndef PFProducer_PFEGammaFilters_H
#define PFProducer_PFEGammaFilters_H

#include "TMath.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>

class PFEGammaFilters {
public:
  PFEGammaFilters(const edm::ParameterSet &iConfig);

  bool passPhotonSelection(const reco::Photon &) const;
  bool passElectronSelection(const reco::GsfElectron &, const reco::PFCandidate &, const int &) const;
  bool isElectron(const reco::GsfElectron &) const;

  bool isElectronSafeForJetMET(const reco::GsfElectron &,
                               const reco::PFCandidate &,
                               const reco::Vertex &,
                               bool &lockTracks) const;

  bool isPhotonSafeForJetMET(const reco::Photon &, const reco::PFCandidate &) const;

  static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

private:
  bool passGsfElePreSelWithOnlyConeHadem(const reco::GsfElectron &) const;

  // Photon selections
  const float ph_Et_;
  const float ph_combIso_;
  const float ph_loose_hoe_;
  const float ph_sietaieta_eb_;
  const float ph_sietaieta_ee_;
  float pho_sumPtTrackIso_;
  float pho_sumPtTrackIsoSlope_;

  // Electron selections
  const float ele_iso_pt_;
  const float ele_iso_mva_eb_;
  const float ele_iso_mva_ee_;
  const float ele_iso_combIso_eb_;
  const float ele_iso_combIso_ee_;
  const float ele_noniso_mva_;
  const unsigned int ele_missinghits_;
  const float ele_ecalDrivenHademPreselCut_;
  const float ele_maxElePtForOnlyMVAPresel_;
  float ele_maxNtracks_;
  float ele_maxHcalE_;
  float ele_maxTrackPOverEele_;
  float ele_maxE_;
  float ele_maxEleHcalEOverEcalE_;
  float ele_maxEcalEOverPRes_;
  float ele_maxEeleOverPoutRes_;
  float ele_maxHcalEOverP_;
  float ele_maxHcalEOverEcalE_;
  float ele_maxEcalEOverP_1_;
  float ele_maxEcalEOverP_2_;
  float ele_maxEeleOverPout_;
  float ele_maxDPhiIN_;

  // dead hcal selections (electrons)
  std::array<float, 2> badHcal_full5x5_sigmaIetaIeta_;
  std::array<float, 2> badHcal_eInvPInv_;
  std::array<float, 2> badHcal_dEta_;
  std::array<float, 2> badHcal_dPhi_;
  bool badHcal_eleEnable_;

  // dead hcal selections (photons)
  float badHcal_phoTrkSolidConeIso_offs_;
  float badHcal_phoTrkSolidConeIso_slope_;
  bool badHcal_phoEnable_;

  const bool debug_ = false;
};
#endif
