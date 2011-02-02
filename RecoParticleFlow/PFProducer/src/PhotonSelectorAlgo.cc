// 
// Original Authors: Nicholas Wardle, Florian Beaudette
//
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"

PhotonSelectorAlgo::PhotonSelectorAlgo(
				       float c_Et,
				       float c_iso_track_a, float c_iso_track_b,
				       float c_iso_ecal_a, float c_iso_ecal_b,
				       float c_iso_hcal_a, float c_iso_hcal_b,
				       float c_hoe
				       ):
  c_Et_(c_Et),
  c_iso_track_a_(c_iso_track_a),  c_iso_track_b_(c_iso_track_b),
  c_iso_ecal_a_(c_iso_ecal_a),  c_iso_ecal_b_(c_iso_ecal_b),
  c_iso_hcal_a_(c_iso_hcal_a), c_iso_hcal_b_(c_iso_hcal_b),
  c_hoe_(c_hoe_)
{
  ;
}

bool PhotonSelectorAlgo::passPhotonSelection(const reco::Photon & photon) const {

  // Photon ET
  float photonPt=photon.pt();
  if( photonPt < c_Et_ ) return false;

  // HoE
  if (photon.hadronicOverEm() > c_hoe_ ) return false;

  // Track iso
  if( photon.trkSumPtHollowConeDR04() > c_iso_track_a_ + c_iso_track_b_*photonPt) return false;

  // ECAL iso
  if (photon.ecalRecHitSumEtConeDR04() > c_iso_ecal_a_ + c_iso_ecal_b_*photonPt) return false;

  // HCAL iso
  if (photon.hcalTowerSumEtConeDR04() > c_iso_hcal_a_ + c_iso_hcal_b_*photonPt) return false ;

  return true;
}
