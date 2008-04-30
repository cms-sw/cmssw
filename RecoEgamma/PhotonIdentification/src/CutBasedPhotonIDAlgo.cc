#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include <iostream>

void CutBasedPhotonIDAlgo::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);

  //get cuts here
  looseEMEcalRecHitIsolationCut_ = conf.getParameter<double>("LooseEMEcalRecHitIso");
  looseEMHcalRecHitIsolationCut_ = conf.getParameter<double>("LooseEMHcalRecHitIso");
  looseEMHollowConeTrkIsolationCut_ = conf.getParameter<double>("LooseEMHollowTrk");
  looseEMSolidConeTrkIsolationCut_ = conf.getParameter<double>("LooseEMSolidTrk");
  looseEMSolidConeNTrkCut_ = conf.getParameter<int>("LooseEMSolidNTrk");
  looseEMHollowConeNTrkCut_ = conf.getParameter<int>("LooseEMHollowNTrk");
  looseEMEtaWidthCut_ = conf.getParameter<double>("LooseEMEtaWidth");
  looseEMHadOverEMCut_ = conf.getParameter<double>("LooseEMHadOverEM");
  looseEMR9Cut_ = conf.getParameter<double>("LooseEMR9Cut");

  loosephotonEcalRecHitIsolationCut_ = conf.getParameter<double>("LoosePhotonEcalRecHitIso");
  loosephotonHcalRecHitIsolationCut_ = conf.getParameter<double>("LoosePhotonHcalRecHitIso");
  loosephotonHollowConeTrkIsolationCut_ = conf.getParameter<double>("LoosePhotonHollowTrk");
  loosephotonSolidConeTrkIsolationCut_ = conf.getParameter<double>("LoosePhotonSolidTrk");
  loosephotonSolidConeNTrkCut_ = conf.getParameter<int>("LoosePhotonSolidNTrk");
  loosephotonHollowConeNTrkCut_ = conf.getParameter<int>("LoosePhotonHollowNTrk");
  loosephotonEtaWidthCut_ = conf.getParameter<double>("LoosePhotonEtaWidth");
  loosephotonHadOverEMCut_ = conf.getParameter<double>("LoosePhotonHadOverEM");
  loosephotonR9Cut_ = conf.getParameter<double>("LoosePhotonR9Cut");

  tightphotonEcalRecHitIsolationCut_ = conf.getParameter<double>("TightPhotonEcalRecHitIso");
  tightphotonHcalRecHitIsolationCut_ = conf.getParameter<double>("TightPhotonHcalRecHitIso");
  tightphotonHollowConeTrkIsolationCut_ = conf.getParameter<double>("TightPhotonHollowTrk");
  tightphotonSolidConeTrkIsolationCut_ = conf.getParameter<double>("TightPhotonSolidTrk");
  tightphotonSolidConeNTrkCut_ = conf.getParameter<int>("TightPhotonSolidNTrk");
  tightphotonHollowConeNTrkCut_ = conf.getParameter<int>("TightPhotonHollowNTrk");
  tightphotonEtaWidthCut_ = conf.getParameter<double>("TightPhotonEtaWidth");
  tightphotonHadOverEMCut_ = conf.getParameter<double>("TightPhotonHadOverEM");
  tightphotonR9Cut_ = conf.getParameter<double>("TightPhotonR9Cut");

  trackConeOuterRadius_ = conf.getParameter<double>("TrackConeOuterRadius");
  trackConeInnerRadius_ = conf.getParameter<double>("TrackConeInnerRadius");
  isolationtrackThreshold_ = conf.getParameter<double>("isolationtrackThreshold");
  photonEcalRecHitConeInnerRadius_ = conf.getParameter<double>("EcalRecHitInnerRadius");
  photonEcalRecHitConeOuterRadius_ = conf.getParameter<double>("EcalRecHitOuterRadius");
  photonEcalRecHitThresh_ = conf.getParameter<double>("EcalRecThresh");
  photonHcalRecHitConeInnerRadius_ = conf.getParameter<double>("HcalRecHitInnerRadius");
  photonHcalRecHitConeOuterRadius_ = conf.getParameter<double>("HcalRecHitOuterRadius");
  photonHcalRecHitThresh_ = conf.getParameter<double>("HcalRecHitThresh");

  //Decision cuts
  dophotonEcalRecHitIsolationCut_ = conf.getParameter<bool>("DoEcalRecHitIsolationCut");
  dophotonHcalRecHitIsolationCut_ = conf.getParameter<bool>("DoHcalRecHitIsolationCut");
  dophotonHCTrkIsolationCut_ = conf.getParameter<bool>("DoHollowConeTrackIsolationCut");
  dophotonSCTrkIsolationCut_ = conf.getParameter<bool>("DoSolidConeTrackIsolationCut");
  dophotonHCNTrkCut_ = conf.getParameter<bool>("DoHollowConeNTrkCut");
  dophotonSCNTrkCut_ = conf.getParameter<bool>("DoSolidConeNTrkCut");
  dophotonHadOverEMCut_ = conf.getParameter<bool>("DoHadOverEMCut");
  dophotonsigmaeeCut_ = conf.getParameter<bool>("DoEtaWidthCut");
  dophotonR9Cut_ = conf.getParameter<bool>("DoR9Cut");
  dorequireNotElectron_ = conf.getParameter<bool>("RequireNotElectron");
  dorequireFiducial_ = conf.getParameter<bool>("RequireFiducial");

}

reco::PhotonID CutBasedPhotonIDAlgo::calculate(const reco::Photon* pho, const edm::Event& e, const edm::EventSetup& es){

  //need to do the following things here:
  //1.)  Call base class methods to calculate photonID variables like fiducial and
  //     isolations.
  //2.)  Decide whether this particular photon passes the cuts that are set forth in the ps.
  //3.)  Create a new PhotonID object, complete with decision and return it.
  
  //  std::cout << "Entering Calculate fcn: " << std::endl;

  //Get fiducial information
  bool isEBPho   = false;
  bool isEEPho   = false;
  bool isEBGap   = false;
  bool isEEGap   = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap);

  std::cout << "Output from classification: " << std::endl;
  std::cout << "Photon Eta: " << pho->p4().Eta();
  std::cout << " Photon phi: " << pho->p4().Phi() << std::endl;
  std::cout << "Flags: ";
  std::cout << "isEBPho: " << isEBPho;
  std::cout << " isEEPho: " << isEEPho;
  std::cout << " isEBGap: " << isEBGap;
  std::cout << " isEEGap: " << isEEGap;
  std::cout << " isEBEEGap: " << isEBEEGap << std::endl;

  //Calculate hollow cone track isolation
  int ntrk=0;
  double trkiso=0;
  calculateTrackIso(pho, e, trkiso, ntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, trackConeInnerRadius_);

  std::cout << "Output from hollow cone track isolation: ";
  std::cout << " Sum pT: " << trkiso << " ntrk: " << ntrk << std::endl;

  //Calculate solid cone track isolation
  int sntrk=0;
  double strkiso=0;
  calculateTrackIso(pho, e, strkiso, sntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, 0.);

  std::cout << "Output from solid cone track isolation: ";
  std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;
  
  double EcalRecHitIso = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadius_,
						photonEcalRecHitConeInnerRadius_,
						photonEcalRecHitThresh_);

  std::cout << "Output from ecal isolation: ";
  std::cout << " Sum pT: " << EcalRecHitIso << std::endl;

  double HcalRecHitIso = calculateHcalRecHitIso(pho, e, es,
						photonHcalRecHitConeOuterRadius_,
						photonHcalRecHitConeInnerRadius_,
						photonHcalRecHitThresh_);

  std::cout << "Output from hcal isolation: ";
  std::cout << " Sum pT: " << HcalRecHitIso << std::endl;

  double EcalR9 = 0;
  //R9 calculation will go HERE.
  //  EcalR9 = calculateR9(pho, e, es);
  //

  bool isElec = isAlsoElectron(pho, e);

  std::cout << "Are you also an electron? " << isElec << std::endl;

  reco::PhotonID temp(false, false, false, strkiso,
		      trkiso, sntrk, ntrk,
		      EcalRecHitIso, HcalRecHitIso, EcalR9,
		      isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap,
		      isElec);

  decide(temp, pho);
  
  std::cout << "Cut based decision: " << temp.isLooseEM() << " " << temp.isLoosePhoton() <<  " " << temp.isTightPhoton() << std::endl;
  
  return temp;

}
void CutBasedPhotonIDAlgo::decide(reco::PhotonID &phID, const reco::Photon* pho){


  ////////////
  //If one has selected to apply fiducial cuts, they will be
  //applied for all looseEM, loosePhoton, tightPhoton.
  //Consider yourself warned!
  ///////////
  //Require that this is not also an Electron supercluster
  if (dorequireNotElectron_){
    if (phID.isAlsoElectron()){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phID.isEBEEGap()) {
      phID.setDecision(false, false, false);
      return;
    }
    if (phID.isEBPho() && phID.isEBGap()){ 
      phID.setDecision(false, false, false);
      return;
    }
    if (phID.isEEPho() && phID.isEEGap()){
      phID.setDecision(false, false, false);
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
    if(phID.isolationEcalRecHit() > looseEMEcalRecHitIsolationCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > looseEMHcalRecHitIsolationCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > looseEMSolidConeNTrkCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > looseEMHollowConeNTrkCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > looseEMSolidConeTrkIsolationCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > looseEMHollowConeTrkIsolationCut_){
      phID.setDecision(false, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (phID.isEEPho()){
      sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    }
    if (sigmaee > looseEMEtaWidthCut_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < looseEMR9Cut_){
      phID.setDecision(false, false, false);
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
    if(phID.isolationEcalRecHit() > loosephotonEcalRecHitIsolationCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > loosephotonHcalRecHitIsolationCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > loosephotonSolidConeNTrkCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > loosephotonHollowConeNTrkCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > loosephotonSolidConeTrkIsolationCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > loosephotonHollowConeTrkIsolationCut_){
      phID.setDecision(true, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (phID.isEEPho()){
      sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    }
    if (sigmaee > loosephotonEtaWidthCut_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < loosephotonR9Cut_){
      phID.setDecision(true, false, false);
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
    if(phID.isolationEcalRecHit() > tightphotonEcalRecHitIsolationCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > tightphotonHcalRecHitIsolationCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > tightphotonSolidConeNTrkCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > tightphotonHollowConeNTrkCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > tightphotonSolidConeTrkIsolationCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > tightphotonHollowConeTrkIsolationCut_){
      phID.setDecision(true, true, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (phID.isEEPho()){
      sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    }
    if (sigmaee > tightphotonEtaWidthCut_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < tightphotonR9Cut_){
      phID.setDecision(true, true, false);
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  phID.setDecision(true, true, true);
  

}
