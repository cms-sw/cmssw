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

#include <iostream>

class PFEGammaFilters {
  
 public:
   
  PFEGammaFilters(float ph_Et,
		  float ph_combIso,
		  float ph_loose_hoe,
		  std::vector<double> & ph_protectionsForJetMET,
		  float ele_iso_pt,
		  float ele_iso_mva_eb,
		  float ele_iso_mva_ee,
		  float ele_iso_combIso_eb,
		  float ele_iso_combIso_ee,
		  float ele_noniso_mva,
		  unsigned int ele_missinghits,
		  std::string ele_iso_path_mvaWeightFile,
		  std::vector<double> & ele_protectionsForJetMET
		  );
  
  ~PFEGammaFilters(){delete ele_iso_mvaID_;};
  
  bool passPhotonSelection(const reco::Photon &);
  bool passElectronSelection(const reco::GsfElectron &, 
			     const reco::PFCandidate &,
			     const int & );
  bool isElectron(const reco::GsfElectron & );
  
  bool isElectronSafeForJetMET(const reco::GsfElectron &, 
			       const reco::PFCandidate &,
			       const reco::Vertex &,
			       bool lockTracks);

  bool isPhotonSafeForJetMET(const reco::Photon &, 
			     const reco::PFCandidate &);

  

 private:

  unsigned int whichTrackAlgo(const reco::TrackRef& trackRef);

  // Photon selections
  float ph_Et_;
  float ph_combIso_;
  float ph_loose_hoe_;
  std::vector<double> ph_protectionsForJetMET_;

  // Electron selections 
  float ele_iso_pt_;
  float ele_iso_mva_eb_;
  float ele_iso_mva_ee_;
  float ele_iso_combIso_eb_;
  float ele_iso_combIso_ee_;
  std::string ele_iso_mva_weightFile_;
  ElectronMVAEstimator *ele_iso_mvaID_;
  float ele_noniso_mva_;
  unsigned int ele_missinghits_;
  std::vector<double> ele_protectionsForJetMET_;
  // Event variables 
  

};
#endif
