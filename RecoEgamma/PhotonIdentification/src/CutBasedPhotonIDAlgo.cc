#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonID.h"
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

void CutBasedPhotonIDAlgo::calculate(const reco::Photon* pho,
				     const edm::Event& e,
				     const edm::EventSetup& es,
				     PhotonFiducialFlags& phofid, 
				     PhotonIsolationVariables& phoisolR1, 
				     PhotonIsolationVariables& phoisolR2, 
				     CutBasedPhotonID &phoid){

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

  phofid.isEBPho = isEBPho;
  phofid.isEEPho = isEEPho;
  phofid.isEBGap = isEBGap;
  phofid.isEEGap = isEEGap;
  phofid.isEBEEGap = isEBEEGap;


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

  phoisolR1.nTrkHollowCone = ntrkA;
  phoisolR1.isolationHollowTrkCone = trkisoA;
  phoisolR1.nTrkSolidCone = sntrkA;
  phoisolR1.isolationSolidTrkCone = strkisoA;

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

  phoisolR2.nTrkHollowCone = ntrkB;
  phoisolR2.isolationHollowTrkCone = trkisoB;
  phoisolR2.nTrkSolidCone = sntrkB;
  phoisolR2.isolationSolidTrkCone = strkisoB;

//   std::cout << "Output from solid cone track isolation: ";
//   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;
  
  double EcalRecHitIsoA = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusA_,
						photonEcalRecHitConeInnerRadiusA_,
                                                photonEcalRecHitEtaSliceA_,
						photonEcalRecHitThreshEA_,
						photonEcalRecHitThreshEtA_);
  phoisolR1.isolationEcalRecHit = EcalRecHitIsoA;

  double EcalRecHitIsoB = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadiusB_,
						photonEcalRecHitConeInnerRadiusB_,
                                                photonEcalRecHitEtaSliceB_,
						photonEcalRecHitThreshEB_,
						photonEcalRecHitThreshEtB_);
  phoisolR2.isolationEcalRecHit = EcalRecHitIsoB;

  double HcalTowerIsoA = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusA_,
					      photonHcalTowerConeInnerRadiusA_,
					      photonHcalTowerThreshEA_);
  phoisolR1.isolationHcalTower = HcalTowerIsoA;

  double HcalTowerIsoB = calculateHcalTowerIso(pho, e, es, photonHcalTowerConeOuterRadiusB_,
					      photonHcalTowerConeInnerRadiusB_,
					      photonHcalTowerThreshEB_);
  phoisolR2.isolationHcalTower = HcalTowerIsoB;

  CutBasedPhotonID phID;
  if (isEBPho)
    decideEB(phofid, phoisolR1, pho, phoid );
  else
    decideEE(phofid, phoisolR1, pho, phoid );
  
  
  //  return temp;

}

void CutBasedPhotonIDAlgo::decideEB(PhotonFiducialFlags phofid, 
				    PhotonIsolationVariables &phoIsolR1, 
				    const reco::Photon* pho,
				    CutBasedPhotonID&  phID ){


  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all looseEM, loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////
  
  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phofid.isEBEEGap) {
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
    if (phofid.isEBPho && phofid.isEBGap){ 
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
    if (phofid.isEEPho && phofid.isEEGap){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
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
    if(phoIsolR1.isolationEcalRecHit > looseEMEcalRecHitIsolationCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > looseEMHcalTowerIsolationCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > looseEMSolidConeNTrkCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > looseEMHollowConeNTrkCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > looseEMSolidConeTrkIsolationCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > looseEMHollowConeTrkIsolationCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }  
  }
  
  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //eta width
  if (dophotonsigmaeeCut_){
    double sigmaee = pho->covIetaIeta();
    if (sigmaee > looseEMEtaWidthCutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < looseEMR9CutEB_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
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
    if(phoIsolR1.isolationEcalRecHit > loosephotonEcalRecHitIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > loosephotonHcalTowerIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > loosephotonSolidConeNTrkCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > loosephotonHollowConeNTrkCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;    
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > loosephotonSolidConeTrkIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > loosephotonHollowConeTrkIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
 
    double sigmaee = pho->covIetaIeta();
    if (sigmaee > loosephotonEtaWidthCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < loosephotonR9CutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;      
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
    if(phoIsolR1.isolationEcalRecHit > tightphotonEcalRecHitIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > tightphotonHcalTowerIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > tightphotonSolidConeNTrkCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > tightphotonHollowConeNTrkCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > tightphotonSolidConeTrkIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > tightphotonHollowConeTrkIsolationCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    double sigmaee = pho->covIetaIeta();
    if (sigmaee > tightphotonEtaWidthCutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < tightphotonR9CutEB_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  phID.isLooseEM = true;
  phID.isLoosePhoton = true;
  phID.isTightPhoton = true;
  
}



void CutBasedPhotonIDAlgo::decideEE(PhotonFiducialFlags phofid, 
				    PhotonIsolationVariables &phoIsolR1, 
				    const reco::Photon* pho, 
				    CutBasedPhotonID&  phID ){
  
  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all looseEM, loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////

  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phofid.isEBEEGap) {
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;

      return;
    }
    if (phofid.isEBPho && phofid.isEBGap){ 
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
    if (phofid.isEEPho && phofid.isEEGap){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
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
    if(phoIsolR1.isolationEcalRecHit > looseEMEcalRecHitIsolationCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;      
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > looseEMHcalTowerIsolationCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }
  
  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > looseEMSolidConeNTrkCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > looseEMHollowConeNTrkCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;     
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > looseEMSolidConeTrkIsolationCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > looseEMHollowConeTrkIsolationCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){   
    double sigmaee = pho->covIetaIeta();
    
    if (sigmaee > looseEMEtaWidthCutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;      
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < looseEMR9CutEE_){
      phID.isLooseEM = false;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;      
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
    if(phoIsolR1.isolationEcalRecHit > loosephotonEcalRecHitIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > loosephotonHcalTowerIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;        
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > loosephotonSolidConeNTrkCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > loosephotonHollowConeNTrkCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > loosephotonSolidConeTrkIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > loosephotonHollowConeTrkIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    
    double sigmaee = pho->covIetaIeta();
  
    if (sigmaee > loosephotonEtaWidthCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < loosephotonR9CutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = false;
      phID.isTightPhoton = false;  
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
    if(phoIsolR1.isolationEcalRecHit > tightphotonEcalRecHitIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalTowerIsolationCut_){
    if(phoIsolR1.isolationHcalTower > tightphotonHcalTowerIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phoIsolR1.nTrkSolidCone > tightphotonSolidConeNTrkCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phoIsolR1.nTrkHollowCone > tightphotonHollowConeNTrkCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phoIsolR1.isolationSolidTrkCone > tightphotonSolidConeTrkIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phoIsolR1.isolationHollowTrkCone > tightphotonHollowConeTrkIsolationCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
   
    double sigmaee = pho->covIetaIeta();
        
    if (sigmaee > tightphotonEtaWidthCutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (pho->r9() < tightphotonR9CutEE_){
      phID.isLooseEM = true;
      phID.isLoosePhoton = true;
      phID.isTightPhoton = false;  
      return;
    }
  }

  //if you got here, you must have passed all cuts!
  phID.isLooseEM = true;
  phID.isLoosePhoton = true;
  phID.isTightPhoton = true;   
  
}
