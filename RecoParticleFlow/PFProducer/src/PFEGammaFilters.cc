// 
// Original Authors: Nicholas Wardle, Florian Beaudette
//
#include "RecoParticleFlow/PFProducer/interface/PFEGammaFilters.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using namespace std;
using namespace reco;

PFEGammaFilters::PFEGammaFilters(float ph_Et,
				 float ph_combIso,
				 float ph_loose_hoe,
				 float ele_iso_pt,
				 float ele_iso_mva_eb,
				 float ele_iso_mva_ee,
				 float ele_iso_combIso_eb,
				 float ele_iso_combIso_ee,
				 float ele_noniso_mva,
				 unsigned int ele_missinghits,
				 string ele_iso_path_mvaWeightFile
				 ):
  ph_Et_(ph_Et),
  ph_combIso_(ph_combIso),
  ph_loose_hoe_(ph_loose_hoe),
  ele_iso_pt_(ele_iso_pt),
  ele_iso_mva_eb_(ele_iso_mva_eb),
  ele_iso_mva_ee_(ele_iso_mva_ee),
  ele_iso_combIso_eb_(ele_iso_combIso_eb),
  ele_iso_combIso_ee_(ele_iso_combIso_ee),
  ele_noniso_mva_(ele_noniso_mva),
  ele_missinghits_(ele_missinghits)
{
  ele_iso_mvaID_= new ElectronMVAEstimator(ele_iso_path_mvaWeightFile);
}

bool PFEGammaFilters::passPhotonSelection(const reco::Photon & photon) {
  // First simple selection, same as the Run1 to be improved in CMSSW_710


  // Photon ET
  if(photon.pt()  < ph_Et_ ) return false;
  //std::cout<<"Cuts "<<combIso_<<" H/E "<<loose_hoe_<<std::endl;
  if (photon.hadronicOverEm() >ph_loose_hoe_ ) return false;
  //Isolation variables in 0.3 cone combined
  if(photon.trkSumPtHollowConeDR03()+photon.ecalRecHitSumEtConeDR03()+photon.hcalTowerSumEtConeDR03() > ph_combIso_)
    return false;		
  
  
  return true;
}
bool PFEGammaFilters::passElectronSelection(const reco::GsfElectron & electron, 
					    const int & nVtx) {
  // First simple selection, same as the Run1 to be improved in CMSSW_710
  
  bool passEleSelection = false;
  
  // Electron ET
  float electronPt = electron.pt();
  
  if( electronPt > ele_iso_pt_) {

    double isoDr03 = electron.dr03TkSumPt() + electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt();
    double eleEta = fabs(electron.eta());
    if (eleEta <= 1.485 && isoDr03 < ele_iso_combIso_eb_) {
      if( ele_iso_mvaID_->mva( electron, nVtx ) > ele_iso_mva_eb_ ) 
	passEleSelection = true;
    }
    else if (eleEta > 1.485  && isoDr03 < ele_iso_combIso_ee_) {
      if( ele_iso_mvaID_->mva( electron, nVtx ) > ele_iso_mva_ee_ ) 
	passEleSelection = true;
    }

  }

//   if(electron.mva() > ele_noniso_mva_) 
//     passEleSelection = true;
 
  
  return passEleSelection;
}

bool PFEGammaFilters::isElectron(const reco::GsfElectron & electron) {
 
  unsigned int nmisshits = electron.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits();
  if(nmisshits > ele_missinghits_)
    return false;

  return true;
}
