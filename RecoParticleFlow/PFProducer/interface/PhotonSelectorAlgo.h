#ifndef PFProducer_PhotonSelectorAlgo_H
#define PFProducer_PhotonSelectorAlgo_H

#include "TMath.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhotonSelectorAlgo {
  
 public:
   
  PhotonSelectorAlgo(float choice,
		     float c_Et_,
		     float c_iso_track_a, float c_iso_track_b,
		     float c_iso_ecal_a, float c_iso_ecal_b,
		     float c_iso_hcal_a, float c_hcal_b,
		     float c_hoe_,
		     float comb_iso,
		     float loose_hoe
		     );
  

  ~PhotonSelectorAlgo(){};
  
  bool passPhotonSelection(const reco::Photon &) const ;
  
 private:
  //Choice of the cuts
  int choice_;
  //First Choice int 0
  // Et cut
    float c_Et_;
  // Track iso, constant term & slope
  float c_iso_track_a_, c_iso_track_b_;
  // ECAL iso, constant term & slope 
  float c_iso_ecal_a_,  c_iso_ecal_b_;
  // HCAL iso, constant term & slope
  float c_iso_hcal_a_,  c_iso_hcal_b_;
  float c_hoe_;
  
  //second choice int 1
  float comb_iso_;
  float loose_hoe_;
};
#endif
