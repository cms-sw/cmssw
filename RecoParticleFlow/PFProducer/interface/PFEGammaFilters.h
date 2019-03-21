#ifndef PFProducer_PFEGammaFilters_H
#define PFProducer_PFEGammaFilters_H

#include "TMath.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class PFEGammaFilters {
  
 public:
   
  PFEGammaFilters(const edm::ParameterSet& iConfig);
  
  bool passPhotonSelection(const reco::Photon &) const;
  bool passElectronSelection(const reco::GsfElectron &, 
			     const reco::PFCandidate &,
			     const int & ) const;
  bool isElectron(const reco::GsfElectron & ) const;
  
  bool isElectronSafeForJetMET(const reco::GsfElectron &, 
			       const reco::PFCandidate &,
			       const reco::Vertex &,
			       bool& lockTracks) const;

  bool isPhotonSafeForJetMET(const reco::Photon &, 
			     const reco::PFCandidate &) const;
  

 private:
  bool passGsfElePreSelWithOnlyConeHadem(const reco::GsfElectron &) const;


  // Photon selections
  const float ph_Et_;
  const float ph_combIso_;
  const float ph_loose_hoe_;
  const float ph_sietaieta_eb_;
  const float ph_sietaieta_ee_;
  const float pho_sumPtTrackIso, pho_sumPtTrackIsoSlope;

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
  const float ele_maxNtracks, ele_maxHcalE, ele_maxTrackPOverEele, ele_maxE,
    ele_maxEleHcalEOverEcalE, ele_maxEcalEOverPRes, ele_maxEeleOverPoutRes,
    ele_maxHcalEOverP, ele_maxHcalEOverEcalE, ele_maxEcalEOverP_1,
    ele_maxEcalEOverP_2, ele_maxEeleOverPout, ele_maxDPhiIN;

  // dead hcal selections (electrons)
  std::array<float,2> badHcal_full5x5_sigmaIetaIeta_;
  std::array<float,2> badHcal_eInvPInv_;
  std::array<float,2> badHcal_dEta_;
  std::array<float,2> badHcal_dPhi_;
  const bool badHcal_eleEnable_;

  // dead hcal selections (photons)
  const float badHcal_phoTrkSolidConeIso_offs_, badHcal_phoTrkSolidConeIso_slope_;
  const bool badHcal_phoEnable_;

  // Event variables 
  
  const bool debug_ = false;
};
#endif
