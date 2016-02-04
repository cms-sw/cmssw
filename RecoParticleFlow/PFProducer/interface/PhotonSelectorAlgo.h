#ifndef PFProducer_PhotonSelectorAlgo_H
#define PFProducer_PhotonSelectorAlgo_H

#include "TMath.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhotonSelectorAlgo {
  
 public:
   
  PhotonSelectorAlgo(float c_Et_,
		     float c_iso_track_a, float c_iso_track_b,
		     float c_iso_ecal_a, float c_iso_ecal_b,
		     float c_iso_hcal_a, float c_hcal_b,
		     float c_hoe_
		     );
  

  ~PhotonSelectorAlgo(){};
  
  bool passPhotonSelection(const reco::Photon &) const ;
  
 private:
  // Et cut
  float c_Et_;
  // Track iso, constant term & slope
  float c_iso_track_a_, c_iso_track_b_;
  // ECAL iso, constant term & slope 
  float c_iso_ecal_a_,  c_iso_ecal_b_;
  // HCAL iso, constant term & slope
  float c_iso_hcal_a_,  c_iso_hcal_b_;
  float c_hoe_;

};
#endif
