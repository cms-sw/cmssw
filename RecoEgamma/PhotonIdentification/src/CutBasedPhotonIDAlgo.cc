#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"



void CutBasedPhotonIDAlgo::setup(const edm::ParameterSet& conf) {
  

  //Decision cuts
  dophotonEcalRecHitIsolationCut_ = conf.getParameter<bool>("DoEcalRecHitIsolationCut");
  dophotonHcalTowerIsolationCut_ = conf.getParameter<bool>("DoHcalTowerIsolationCut");
  dophotonHCTrkIsolationCut_ = conf.getParameter<bool>("DoHollowConeTrackIsolationCut");
  dophotonSCTrkIsolationCut_ = conf.getParameter<bool>("DoSolidConeTrackIsolationCut");
  dophotonHCNTrkCut_ = conf.getParameter<bool>("DoHollowConeNTrkCut");
  dophotonSCNTrkCut_ = conf.getParameter<bool>("DoSolidConeNTrkCut");
  dophotonHadOverEMCut_ = conf.getParameter<bool>("DoHadOverEMCut");
  dophotonsigmaeeCut_ = conf.getParameter<bool>("DoEtaWidthCut");
  dophotonR9Cut_ = conf.getParameter<bool>("DoR9Cut");
  dorequireFiducial_ = conf.getParameter<bool>("RequireFiducial");

  loosephotonEcalRecHitIsolationCutEB_ = conf.getParameter<double>("LoosePhotonEcalRecHitIsoEB");
  loosephotonHcalTowerIsolationCutEB_ = conf.getParameter<double>("LoosePhotonHcalTowerIsoEB");
  loosephotonHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("LoosePhotonHollowTrkEB");
  loosephotonSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("LoosePhotonSolidTrkEB");
  loosephotonSolidConeNTrkCutEB_ = conf.getParameter<int>("LoosePhotonSolidNTrkEB");
  loosephotonHollowConeNTrkCutEB_ = conf.getParameter<int>("LoosePhotonHollowNTrkEB");
  loosephotonEtaWidthCutEB_ = conf.getParameter<double>("LoosePhotonEtaWidthEB");
  loosephotonHadOverEMCutEB_ = conf.getParameter<double>("LoosePhotonHadOverEMEB");
  loosephotonR9CutEB_ = conf.getParameter<double>("LoosePhotonR9CutEB");

  tightphotonEcalRecHitIsolationCutEB_ = conf.getParameter<double>("TightPhotonEcalRecHitIsoEB");
  tightphotonHcalTowerIsolationCutEB_ = conf.getParameter<double>("TightPhotonHcalTowerIsoEB");
  tightphotonHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("TightPhotonHollowTrkEB");
  tightphotonSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("TightPhotonSolidTrkEB");
  tightphotonSolidConeNTrkCutEB_ = conf.getParameter<int>("TightPhotonSolidNTrkEB");
  tightphotonHollowConeNTrkCutEB_ = conf.getParameter<int>("TightPhotonHollowNTrkEB");
  tightphotonEtaWidthCutEB_ = conf.getParameter<double>("TightPhotonEtaWidthEB");
  tightphotonHadOverEMCutEB_ = conf.getParameter<double>("TightPhotonHadOverEMEB");
  tightphotonR9CutEB_ = conf.getParameter<double>("TightPhotonR9CutEB");

  //get cuts here EE
  loosephotonEcalRecHitIsolationCutEE_ = conf.getParameter<double>("LoosePhotonEcalRecHitIsoEE");
  loosephotonHcalTowerIsolationCutEE_ = conf.getParameter<double>("LoosePhotonHcalTowerIsoEE");
  loosephotonHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("LoosePhotonHollowTrkEE");
  loosephotonSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("LoosePhotonSolidTrkEE");
  loosephotonSolidConeNTrkCutEE_ = conf.getParameter<int>("LoosePhotonSolidNTrkEE");
  loosephotonHollowConeNTrkCutEE_ = conf.getParameter<int>("LoosePhotonHollowNTrkEE");
  loosephotonEtaWidthCutEE_ = conf.getParameter<double>("LoosePhotonEtaWidthEE");
  loosephotonHadOverEMCutEE_ = conf.getParameter<double>("LoosePhotonHadOverEMEE");
  loosephotonR9CutEE_ = conf.getParameter<double>("LoosePhotonR9CutEE");

  tightphotonEcalRecHitIsolationCutEE_ = conf.getParameter<double>("TightPhotonEcalRecHitIsoEE");
  tightphotonHcalTowerIsolationCutEE_ = conf.getParameter<double>("TightPhotonHcalTowerIsoEE");
  tightphotonHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("TightPhotonHollowTrkEE");
  tightphotonSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("TightPhotonSolidTrkEE");
  tightphotonSolidConeNTrkCutEE_ = conf.getParameter<int>("TightPhotonSolidNTrkEE");
  tightphotonHollowConeNTrkCutEE_ = conf.getParameter<int>("TightPhotonHollowNTrkEE");
  tightphotonEtaWidthCutEE_ = conf.getParameter<double>("TightPhotonEtaWidthEE");
  tightphotonHadOverEMCutEE_ = conf.getParameter<double>("TightPhotonHadOverEMEE");
  tightphotonR9CutEE_ = conf.getParameter<double>("TightPhotonR9CutEE");

}

void CutBasedPhotonIDAlgo::decideEB(const reco::Photon* pho, bool &LoosePhoton, bool &TightPhoton){


  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////
  
  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (pho->isEBEEGap()) {
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
    if (pho->isEB() && pho->isEBGap()){ 
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
    if (pho->isEE() && pho->isEEGap()){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }
  //////////////
  //Done with fiducial cuts.
  //////////////

  //////////////
  //Next do loosephoton selection.
  //If an object is not LoosePhoton, it is also not
  //TightPhoton!
  //////////////
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(pho->ecalRecHitSumEtConeDR04() > loosephotonEcalRecHitIsolationCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(pho->hcalTowerSumEtConeDR04() > loosephotonHcalTowerIsolationCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (pho->nTrkSolidConeDR04() > loosephotonSolidConeNTrkCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (pho->nTrkHollowConeDR04() > loosephotonHollowConeNTrkCutEB_){
      LoosePhoton = false;
      TightPhoton = false;    
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (pho->trkSumPtSolidConeDR04() > loosephotonSolidConeTrkIsolationCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (pho->trkSumPtHollowConeDR04() > loosephotonHollowConeTrkIsolationCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
 
    double sigmaee = pho->sigmaIetaIeta();
    if (sigmaee > loosephotonEtaWidthCutEB_){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < loosephotonR9CutEB_){
      LoosePhoton = false;
      TightPhoton = false;      
      return;
    }
  }
  //If one reaches this point, the decision has been made that this object,
  //is indeed loosePhoton.

  //////////////
  //Next do tightphoton selection.
  //This is the tightest critieria,
  //and once more one assumes that these criteria are
  //tighter than loose.
  //////////////
    //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(pho->ecalRecHitSumEtConeDR04() > tightphotonEcalRecHitIsolationCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(pho->hcalTowerSumEtConeDR04() > tightphotonHcalTowerIsolationCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (pho->nTrkSolidConeDR04() > tightphotonSolidConeNTrkCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (pho->nTrkHollowConeDR04() > tightphotonHollowConeNTrkCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (pho->trkSumPtSolidConeDR04() > tightphotonSolidConeTrkIsolationCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (pho->trkSumPtHollowConeDR04() > tightphotonHollowConeTrkIsolationCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    double sigmaee = pho->sigmaIetaIeta();
    if (sigmaee > tightphotonEtaWidthCutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < tightphotonR9CutEB_){
      LoosePhoton = true;
      TightPhoton = false;
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  LoosePhoton = true;
  TightPhoton = true;
  
}



void CutBasedPhotonIDAlgo::decideEE(const reco::Photon* pho, bool &LoosePhoton, bool &TightPhoton){
  
  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all , loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////

  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (pho->isEBEEGap()) {
       LoosePhoton = false;
      TightPhoton = false;

      return;
    }
    if (pho->isEB() && pho->isEBGap()){ 
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
    if (pho->isEE() && pho->isEEGap()){
      LoosePhoton = false;
      TightPhoton = false;
      return;
    }
  }
  //////////////
  //Done with fiducial cuts.
  //////////////

  //////////////
  //Next do loosephoton selection.
  //If an object is not LoosePhoton, it is also not
  //TightPhoton!
  //////////////
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(pho->ecalRecHitSumEtConeDR04() > loosephotonEcalRecHitIsolationCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(pho->hcalTowerSumEtConeDR04() > loosephotonHcalTowerIsolationCutEE_){
      LoosePhoton = false;
      TightPhoton = false;        
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (pho->nTrkSolidConeDR04() > loosephotonSolidConeNTrkCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (pho->nTrkHollowConeDR04() > loosephotonHollowConeNTrkCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (pho->trkSumPtSolidConeDR04() > loosephotonSolidConeTrkIsolationCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (pho->trkSumPtHollowConeDR04() > loosephotonHollowConeTrkIsolationCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    
    double sigmaee = pho->sigmaIetaIeta();
  
    if (sigmaee > loosephotonEtaWidthCutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < loosephotonR9CutEE_){
      LoosePhoton = false;
      TightPhoton = false;  
      return;
    }
  }
  //If one reaches this point, the decision has been made that this object,
  //is indeed loosePhoton.

  //////////////
  //Next do tightphoton selection.
  //This is the tightest critieria,
  //and once more one assumes that these criteria are
  //tighter than loose.
  //////////////
    //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(pho->ecalRecHitSumEtConeDR04() > tightphotonEcalRecHitIsolationCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(pho->hcalTowerSumEtConeDR04() > tightphotonHcalTowerIsolationCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (pho->nTrkSolidConeDR04() > tightphotonSolidConeNTrkCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (pho->nTrkHollowConeDR04() > tightphotonHollowConeNTrkCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (pho->trkSumPtSolidConeDR04() > tightphotonSolidConeTrkIsolationCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (pho->trkSumPtHollowConeDR04() > tightphotonHollowConeTrkIsolationCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
   
    double sigmaee = pho->sigmaIetaIeta();
        
    if (sigmaee > tightphotonEtaWidthCutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < tightphotonR9CutEE_){
      LoosePhoton = true;
      TightPhoton = false;  
      return;
    }
  }

  //if you got here, you must have passed all cuts!
  LoosePhoton = true;
  TightPhoton = true;   
  
}
