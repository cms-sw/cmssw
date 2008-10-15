#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"


void CutBasedPhotonIDAlgo::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);

  trackConeOuterRadiusA_ = conf.getParameter<double>("TrackConeOuterRadiusA");
  trackConeInnerRadiusA_ = conf.getParameter<double>("TrackConeInnerRadiusA");
  isolationtrackThresholdA_ = conf.getParameter<double>("isolationtrackThresholdA");

  photonEcalRecHitConeInnerRadiusA_ = conf.getParameter<double>("EcalRecHitInnerRadiusA");
  photonEcalRecHitConeOuterRadiusA_ = conf.getParameter<double>("EcalRecHitOuterRadiusA");
  photonEcalRecHitEtaSliceA_ = conf.getParameter<double>("EcalRecHitEtaSliceA");
  photonEcalRecHitThreshEA_ = conf.getParameter<double>("EcalRecThreshEA");
  photonEcalRecHitThreshEtA_ = conf.getParameter<double>("EcalRecThreshEtA");

  photonHcalTowerConeInnerRadiusA_ = conf.getParameter<double>("HcalTowerInnerRadiusA");
  photonHcalTowerConeOuterRadiusA_ = conf.getParameter<double>("HcalTowerOuterRadiusA");
  photonHcalTowerThreshEA_ = conf.getParameter<double>("HcalTowerThreshEA");

  trackConeOuterRadiusB_ = conf.getParameter<double>("TrackConeOuterRadiusB");
  trackConeInnerRadiusB_ = conf.getParameter<double>("TrackConeInnerRadiusB");
  isolationtrackThresholdB_ = conf.getParameter<double>("isolationtrackThresholdB");

  photonEcalRecHitConeInnerRadiusB_ = conf.getParameter<double>("EcalRecHitInnerRadiusB");
  photonEcalRecHitConeOuterRadiusB_ = conf.getParameter<double>("EcalRecHitOuterRadiusB");
  photonEcalRecHitEtaSliceB_ = conf.getParameter<double>("EcalRecHitEtaSliceB");
  photonEcalRecHitThreshEB_ = conf.getParameter<double>("EcalRecThreshEB");
  photonEcalRecHitThreshEtB_ = conf.getParameter<double>("EcalRecThreshEtB");

  photonHcalTowerConeInnerRadiusB_ = conf.getParameter<double>("HcalTowerInnerRadiusB");
  photonHcalTowerConeOuterRadiusB_ = conf.getParameter<double>("HcalTowerOuterRadiusB");
  photonHcalTowerThreshEB_ = conf.getParameter<double>("HcalTowerThreshEB");

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

  //get cuts here EB first
  looseEMEcalRecHitIsolationCutEB_ = conf.getParameter<double>("LooseEMEcalRecHitIsoEB");
  looseEMHcalTowerIsolationCutEB_ = conf.getParameter<double>("LooseEMHcalTowerIsoEB");
  looseEMHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("LooseEMHollowTrkEB");
  looseEMSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("LooseEMSolidTrkEB");
  looseEMSolidConeNTrkCutEB_ = conf.getParameter<int>("LooseEMSolidNTrkEB");
  looseEMHollowConeNTrkCutEB_ = conf.getParameter<int>("LooseEMHollowNTrkEB");
  looseEMEtaWidthCutEB_ = conf.getParameter<double>("LooseEMEtaWidthEB");
  looseEMHadOverEMCutEB_ = conf.getParameter<double>("LooseEMHadOverEMEB");
  looseEMR9CutEB_ = conf.getParameter<double>("LooseEMR9CutEB");

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
  looseEMEcalRecHitIsolationCutEE_ = conf.getParameter<double>("LooseEMEcalRecHitIsoEE");
  looseEMHcalTowerIsolationCutEE_ = conf.getParameter<double>("LooseEMHcalTowerIsoEE");
  looseEMHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("LooseEMHollowTrkEE");
  looseEMSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("LooseEMSolidTrkEE");
  looseEMSolidConeNTrkCutEE_ = conf.getParameter<int>("LooseEMSolidNTrkEE");
  looseEMHollowConeNTrkCutEE_ = conf.getParameter<int>("LooseEMHollowNTrkEE");
  looseEMEtaWidthCutEE_ = conf.getParameter<double>("LooseEMEtaWidthEE");
  looseEMHadOverEMCutEE_ = conf.getParameter<double>("LooseEMHadOverEMEE");
  looseEMR9CutEE_ = conf.getParameter<double>("LooseEMR9CutEE");

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

void CutBasedPhotonIDAlgo::calculate(const reco::Photon* pho, const edm::Event& e, const edm::EventSetup& es, CutBasedPhotonQuantities &phoid){

  //need to do the following things here:
  //1.)  Call base class methods to calculate photonID variables like fiducial and
  //     isolations.
  //2.)  Decide whether this particular photon passes the cuts that are set forth in the ps.
  //3.)  Set the struct values
  

  //Get fiducial information
  bool isEBPho   = false;
  bool isEEPho   = false;
  bool isEBGap   = false;
  bool isEEGap   = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap);

  phoid.isEBPho_ = isEBPho;
  phoid.isEEPho_ = isEEPho;
  phoid.isEBGap_ = isEBGap;
  phoid.isEEGap_ = isEEGap;
  phoid.isEBEEGap_ = isEBEEGap;


  //Calculate hollow cone track isolation, CONE A
  int ntrkA=0;
  double trkisoA=0;
  calculateTrackIso(pho, e, trkisoA, ntrkA, isolationtrackThresholdA_,    
		    trackConeOuterRadiusA_, trackConeInnerRadiusA_);

  //Calculate solid cone track isolation, CONE A
  int sntrkA=0;
  double strkisoA=0;
  calculateTrackIso(pho, e, strkisoA, sntrkA, isolationtrackThresholdA_,    
		    trackConeOuterRadiusA_, 0.);

  phoid.nTrkHollowConeA_ = ntrkA;
  phoid.isolationHollowTrkConeA_ = trkisoA;
  phoid.nTrkSolidConeA_ = sntrkA;
  phoid.isolationSolidTrkConeA_ = strkisoA;

  //Calculate hollow cone track isolation, CONE B
  int ntrkB=0;
  double trkisoB=0;
  calculateTrackIso(pho, e, trkisoB, ntrkB, isolationtrackThresholdB_,    
		    trackConeOuterRadiusB_, trackConeInnerRadiusB_);

  //Calculate solid cone track isolation, CONE B
  int sntrkB=0;
  double strkisoB=0;
  calculateTrackIso(pho, e, strkisoB, sntrkB, isolationtrackThresholdB_,    
		    trackConeOuterRadiusB_, 0.);

  phoid.nTrkHollowConeB_ = ntrkB;
  phoid.isolationHollowTrkConeB_ = trkisoB;
  phoid.nTrkSolidConeB_ = sntrkB;
  phoid.isolationSolidTrkConeB_ = strkisoB;

//   std::cout << "Output from solid cone track isolation: ";
//   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;
  
  double EcalRecHitIsoA = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusA_,
						photonEcalRecHitConeInnerRadiusA_,
                                                photonEcalRecHitEtaSliceA_,
						photonEcalRecHitThreshEA_,
						photonEcalRecHitThreshEtA_);
  phoid.isolationEcalRecHitA_ = EcalRecHitIsoA;

  double EcalRecHitIsoB = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusB_,
						photonEcalRecHitConeInnerRadiusB_,
                                                photonEcalRecHitEtaSliceB_,
						photonEcalRecHitThreshEB_,
						photonEcalRecHitThreshEtB_);
  phoid.isolationEcalRecHitB_ = EcalRecHitIsoB;

  double HcalTowerIsoA = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusA_,
					      photonHcalTowerConeInnerRadiusA_,
					      photonHcalTowerThreshEA_);
  phoid.isolationHcalTowerA_ = HcalTowerIsoA;

  double HcalTowerIsoB = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusB_,
					      photonHcalTowerConeInnerRadiusB_,
					      photonHcalTowerThreshEB_);
  phoid.isolationHcalTowerB_ = HcalTowerIsoB;

  double r9 = 0;
  r9 = calculateR9(pho, e, es);
  phoid.r9_ = r9;

  double E1x5 = 0;
  E1x5 = calculateE1x5(pho, e, es);
  phoid.e1x5_ = E1x5;

  double E2x5 = 0;
  E2x5 = calculateE2x5(pho, e, es);
  phoid.e2x5_ = E2x5;

  double sigmaIetaIeta=0;
  sigmaIetaIeta = calculateSigmaIetaIeta(pho, e, es);
  phoid.sigmaIetaIeta_ = sigmaIetaIeta;

//   reco::PhotonID temp(false, false, false, strkiso,
// 		      trkiso, sntrk, ntrk,
// 		      EcalRecHitIso, HcalTowerIso, 
// 		      EcalR9, E1x5, E2x5, E5x5, sigmaIetaIeta,
// 		      isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap,
// 		      isElec);
  if (isEBPho)
    decideEB(phoid, pho);
  else
    decideEE(phoid, pho);
  
  //  std::cout << "Cut based decision: " << temp.isLooseEM() << " " << temp.isLoosePhoton() <<  " " << temp.isTightPhoton() << std::endl;
  
  //  return temp;

}
void CutBasedPhotonIDAlgo::decideEB(CutBasedPhotonQuantities &phID, const reco::Photon* pho){


  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all looseEM, loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////
  
  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phID.isEBEEGap_) {
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
    if (phID.isEBPho_ && phID.isEBGap_){ 
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
    if (phID.isEEPho_ && phID.isEEGap_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  //////////////
  //Done with fiducial cuts.
  //////////////

  //////////////
  //First do looseEM selection.
  //If an object is not LooseEM, it is also not
  //LoosePhoton or TightPhoton!
  //////////////
    
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(phID.isolationEcalRecHitA_ > looseEMEcalRecHitIsolationCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > looseEMHcalTowerIsolationCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > looseEMSolidConeNTrkCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > looseEMHollowConeNTrkCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > looseEMSolidConeTrkIsolationCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > looseEMHollowConeTrkIsolationCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }  
  }
  
  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //eta width
  if (dophotonsigmaeeCut_){
    double sigmaee = phID.sigmaIetaIeta_;
    if (sigmaee > looseEMEtaWidthCutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < looseEMR9CutEB_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //If one reaches this point, the decision has been made that this object,
  //is indeed looseEM.

  //////////////
  //Next do loosephoton selection.
  //If an object is not LoosePhoton, it is also not
  //TightPhoton!
  //////////////
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(phID.isolationEcalRecHitA_ > loosephotonEcalRecHitIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > loosephotonHcalTowerIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > loosephotonSolidConeNTrkCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > loosephotonHollowConeNTrkCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;    
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > loosephotonSolidConeTrkIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > loosephotonHollowConeTrkIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
 
    double sigmaee = phID.sigmaIetaIeta_;
    if (sigmaee > loosephotonEtaWidthCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < loosephotonR9CutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;      
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
    if(phID.isolationEcalRecHitA_ > tightphotonEcalRecHitIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > tightphotonHcalTowerIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > tightphotonSolidConeNTrkCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > tightphotonHollowConeNTrkCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > tightphotonSolidConeTrkIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > tightphotonHollowConeTrkIsolationCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    double sigmaee = phID.sigmaIetaIeta_;
    if (sigmaee > tightphotonEtaWidthCutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < tightphotonR9CutEB_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  phID.isLooseEM_ = true;
  phID.isLoosePhoton_ = true;
  phID.isTightPhoton_ = true;
  
}



void CutBasedPhotonIDAlgo::decideEE(CutBasedPhotonQuantities &phID, const reco::Photon* pho){


  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all looseEM, loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////

  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phID.isEBEEGap_) {
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;

      return;
    }
    if (phID.isEBPho_ && phID.isEBGap_){ 
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
    if (phID.isEEPho_ && phID.isEEGap_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  //////////////
  //Done with fiducial cuts.
  //////////////

  //////////////
  //First do looseEM selection.
  //If an object is not LooseEM, it is also not
  //LoosePhoton or TightPhoton!
  //////////////
    
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(phID.isolationEcalRecHitA_ > looseEMEcalRecHitIsolationCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;      
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > looseEMHcalTowerIsolationCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }
  
  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > looseEMSolidConeNTrkCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > looseEMHollowConeNTrkCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;     
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > looseEMSolidConeTrkIsolationCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > looseEMHollowConeTrkIsolationCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){   
    double sigmaee = phID.sigmaIetaIeta_;
    
    if (sigmaee > looseEMEtaWidthCutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;      
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < looseEMR9CutEE_){
      phID.isLooseEM_ = false;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;      
      return;
    }
  }
  
  //If one reaches this point, the decision has been made that this object,
  //is indeed looseEM.

  //////////////
  //Next do loosephoton selection.
  //If an object is not LoosePhoton, it is also not
  //TightPhoton!
  //////////////
  //Cut on the sum of ecal rec hits in a cone
  if(dophotonEcalRecHitIsolationCut_){
    if(phID.isolationEcalRecHitA_ > loosephotonEcalRecHitIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > loosephotonHcalTowerIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;        
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > loosephotonSolidConeNTrkCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > loosephotonHollowConeNTrkCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > loosephotonSolidConeTrkIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > loosephotonHollowConeTrkIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    
    double sigmaee = phID.sigmaIetaIeta_;
  
    if (sigmaee > loosephotonEtaWidthCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < loosephotonR9CutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = false;
      phID.isTightPhoton_ = false;  
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
    if(phID.isolationEcalRecHitA_ > tightphotonEcalRecHitIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phID.isolationHcalTowerA_ > tightphotonHcalTowerIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidConeA_ > tightphotonSolidConeNTrkCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowConeA_ > tightphotonHollowConeNTrkCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkConeA_ > tightphotonSolidConeTrkIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkConeA_ > tightphotonHollowConeTrkIsolationCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
   
    double sigmaee = phID.sigmaIetaIeta_;
        
    if (sigmaee > tightphotonEtaWidthCutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9_ < tightphotonR9CutEE_){
      phID.isLooseEM_ = true;
      phID.isLoosePhoton_ = true;
      phID.isTightPhoton_ = false;  
      return;
    }
  }

  //if you got here, you must have passed all cuts!
  phID.isLooseEM_ = true;
  phID.isLoosePhoton_ = true;
  phID.isTightPhoton_ = true;   
  
}
