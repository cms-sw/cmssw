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
   
  PFEGammaFilters(float ph_Et,
		  float ph_combIso,
		  float ph_loose_hoe,
		  float ph_sietaieta_eb,
		  float ph_sietaieta_ee,
		  const edm::ParameterSet& ph_protectionsForJetMET,
		  const edm::ParameterSet& ph_protectionsForBadHcal,
		  float ele_iso_pt,
		  float ele_iso_mva_eb,
		  float ele_iso_mva_ee,
		  float ele_iso_combIso_eb,
		  float ele_iso_combIso_ee,
		  float ele_noniso_mva,
		  unsigned int ele_missinghits,
		  const std::string& ele_iso_path_mvaWeightFile,
		  const edm::ParameterSet& ele_protectionsForJetMET,
		  const edm::ParameterSet& ele_protectionsForBadHcal
		  );
  
  ~PFEGammaFilters(){};
  
  bool passPhotonSelection(const reco::Photon &);
  bool passElectronSelection(const reco::GsfElectron &, 
			     const reco::PFCandidate &,
			     const int & );
  bool isElectron(const reco::GsfElectron & );
  
  bool isElectronSafeForJetMET(const reco::GsfElectron &, 
			       const reco::PFCandidate &,
			       const reco::Vertex &,
			       bool& lockTracks);

  bool isPhotonSafeForJetMET(const reco::Photon &, 
			     const reco::PFCandidate &);

  void setDebug( bool debug ) { debug_ = debug; }
  

 private:



  // Photon selections
  float ph_Et_;
  float ph_combIso_;
  float ph_loose_hoe_;
  float ph_sietaieta_eb_;
  float ph_sietaieta_ee_;
  //std::vector<double> ph_protectionsForJetMET_; //replacement below
  float pho_sumPtTrackIso, pho_sumPtTrackIsoSlope;

  // Electron selections 
  float ele_iso_pt_;
  float ele_iso_mva_eb_;
  float ele_iso_mva_ee_;
  float ele_iso_combIso_eb_;
  float ele_iso_combIso_ee_;
  float ele_noniso_mva_;
  unsigned int ele_missinghits_;
  //std::vector<double> ele_protectionsForJetMET_; // replacement below
  float ele_maxNtracks, ele_maxHcalE, ele_maxTrackPOverEele, ele_maxE,
    ele_maxEleHcalEOverEcalE, ele_maxEcalEOverPRes, ele_maxEeleOverPoutRes,
    ele_maxHcalEOverP, ele_maxHcalEOverEcalE, ele_maxEcalEOverP_1,
    ele_maxEcalEOverP_2, ele_maxEeleOverPout, ele_maxDPhiIN;

  // dead hcal selections (electrons)
  std::array<float,2> badHcal_full5x5_sigmaIetaIeta_;
  std::array<float,2> badHcal_eInvPInv_;
  std::array<float,2> badHcal_dEta_;
  std::array<float,2> badHcal_dPhi_;
  bool badHcal_eleEnable_;
  static void readEBEEParams_(const edm::ParameterSet &pset, const std::string &name, std::array<float,2> & out) ;

  // dead hcal selections (photons)
  float badHcal_phoTrkSolidConeIso_offs_, badHcal_phoTrkSolidConeIso_slope_;
  bool badHcal_phoEnable_;

  // Event variables 
  
  bool debug_;
};
#endif
