// 
// Original Authors: Nicholas Wardle, Florian Beaudette
//
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"

PhotonSelectorAlgo::PhotonSelectorAlgo(
				       float choice,
				       float c_Et,
				       float c_iso_track_a, float c_iso_track_b,
				       float c_iso_ecal_a, float c_iso_ecal_b,
				       float c_iso_hcal_a, float c_iso_hcal_b,
				       float c_hoe,
				       float comb_iso,
				       float loose_hoe
				       ):
  choice_(choice),
  c_Et_(c_Et),
  c_iso_track_a_(c_iso_track_a),  c_iso_track_b_(c_iso_track_b),
  c_iso_ecal_a_(c_iso_ecal_a),  c_iso_ecal_b_(c_iso_ecal_b),
  c_iso_hcal_a_(c_iso_hcal_a), c_iso_hcal_b_(c_iso_hcal_b),
  c_hoe_(c_hoe),
  comb_iso_(comb_iso),
  loose_hoe_(loose_hoe)
{
  ;
}

bool PhotonSelectorAlgo::passPhotonSelection(const reco::Photon & photon) const {

  // Photon ET
  float photonPt=photon.pt();
  if( photonPt < c_Et_ ) return false;
  if(choice_<0.1) //EGM Loose
    {
      //std::cout<<"Cuts:"<<c_Et_<<" H/E "<<c_hoe_<<"ECal Iso "<<c_iso_ecal_a_<<"HCal Iso "<<c_iso_hcal_a_<<"Track Iso "<<c_iso_track_a_<<std::endl;
      // HoE
      if (photon.hadronicOverEm() > c_hoe_ ) return false;
      
      // Track iso
      if( photon.trkSumPtHollowConeDR04() > c_iso_track_a_ + c_iso_track_b_*photonPt) return false;
      
      // ECAL iso
      if (photon.ecalRecHitSumEtConeDR04() > c_iso_ecal_a_ + c_iso_ecal_b_*photonPt) return false;
      
      // HCAL iso
      if (photon.hcalTowerSumEtConeDR04() > c_iso_hcal_a_ + c_iso_hcal_b_*photonPt) return false ;
    }
  if(choice_>0.99)
    {
      
      //std::cout<<"Cuts "<<comb_iso_<<" H/E "<<loose_hoe_<<std::endl;
      if (photon.hadronicOverEm() >loose_hoe_ ) return false;
      //Isolation variables in 0.3 cone combined
        if(photon.trkSumPtHollowConeDR03()+photon.ecalRecHitSumEtConeDR03()+photon.hcalTowerSumEtConeDR03()>comb_iso_)return false;		
    }

  return true;
}
