#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"


void CutBasedPhotonIDAlgo::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);

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

  //get cuts here EB first
  looseEMEcalRecHitIsolationCutEB_ = conf.getParameter<double>("LooseEMEcalRecHitIsoEB");
  looseEMHcalRecHitIsolationCutEB_ = conf.getParameter<double>("LooseEMHcalRecHitIsoEB");
  looseEMHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("LooseEMHollowTrkEB");
  looseEMSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("LooseEMSolidTrkEB");
  looseEMSolidConeNTrkCutEB_ = conf.getParameter<int>("LooseEMSolidNTrkEB");
  looseEMHollowConeNTrkCutEB_ = conf.getParameter<int>("LooseEMHollowNTrkEB");
  looseEMEtaWidthCutEB_ = conf.getParameter<double>("LooseEMEtaWidthEB");
  looseEMHadOverEMCutEB_ = conf.getParameter<double>("LooseEMHadOverEMEB");
  looseEMR9CutEB_ = conf.getParameter<double>("LooseEMR9CutEB");

  loosephotonEcalRecHitIsolationCutEB_ = conf.getParameter<double>("LoosePhotonEcalRecHitIsoEB");
  loosephotonHcalRecHitIsolationCutEB_ = conf.getParameter<double>("LoosePhotonHcalRecHitIsoEB");
  loosephotonHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("LoosePhotonHollowTrkEB");
  loosephotonSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("LoosePhotonSolidTrkEB");
  loosephotonSolidConeNTrkCutEB_ = conf.getParameter<int>("LoosePhotonSolidNTrkEB");
  loosephotonHollowConeNTrkCutEB_ = conf.getParameter<int>("LoosePhotonHollowNTrkEB");
  loosephotonEtaWidthCutEB_ = conf.getParameter<double>("LoosePhotonEtaWidthEB");
  loosephotonHadOverEMCutEB_ = conf.getParameter<double>("LoosePhotonHadOverEMEB");
  loosephotonR9CutEB_ = conf.getParameter<double>("LoosePhotonR9CutEB");

  tightphotonEcalRecHitIsolationCutEB_ = conf.getParameter<double>("TightPhotonEcalRecHitIsoEB");
  tightphotonHcalRecHitIsolationCutEB_ = conf.getParameter<double>("TightPhotonHcalRecHitIsoEB");
  tightphotonHollowConeTrkIsolationCutEB_ = conf.getParameter<double>("TightPhotonHollowTrkEB");
  tightphotonSolidConeTrkIsolationCutEB_ = conf.getParameter<double>("TightPhotonSolidTrkEB");
  tightphotonSolidConeNTrkCutEB_ = conf.getParameter<int>("TightPhotonSolidNTrkEB");
  tightphotonHollowConeNTrkCutEB_ = conf.getParameter<int>("TightPhotonHollowNTrkEB");
  tightphotonEtaWidthCutEB_ = conf.getParameter<double>("TightPhotonEtaWidthEB");
  tightphotonHadOverEMCutEB_ = conf.getParameter<double>("TightPhotonHadOverEMEB");
  tightphotonR9CutEB_ = conf.getParameter<double>("TightPhotonR9CutEB");

  //get cuts here EE
  looseEMEcalRecHitIsolationCutEE_ = conf.getParameter<double>("LooseEMEcalRecHitIsoEE");
  looseEMHcalRecHitIsolationCutEE_ = conf.getParameter<double>("LooseEMHcalRecHitIsoEE");
  looseEMHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("LooseEMHollowTrkEE");
  looseEMSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("LooseEMSolidTrkEE");
  looseEMSolidConeNTrkCutEE_ = conf.getParameter<int>("LooseEMSolidNTrkEE");
  looseEMHollowConeNTrkCutEE_ = conf.getParameter<int>("LooseEMHollowNTrkEE");
  looseEMEtaWidthCutEE_ = conf.getParameter<double>("LooseEMEtaWidthEE");
  looseEMHadOverEMCutEE_ = conf.getParameter<double>("LooseEMHadOverEMEE");
  looseEMR9CutEE_ = conf.getParameter<double>("LooseEMR9CutEE");

  loosephotonEcalRecHitIsolationCutEE_ = conf.getParameter<double>("LoosePhotonEcalRecHitIsoEE");
  loosephotonHcalRecHitIsolationCutEE_ = conf.getParameter<double>("LoosePhotonHcalRecHitIsoEE");
  loosephotonHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("LoosePhotonHollowTrkEE");
  loosephotonSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("LoosePhotonSolidTrkEE");
  loosephotonSolidConeNTrkCutEE_ = conf.getParameter<int>("LoosePhotonSolidNTrkEE");
  loosephotonHollowConeNTrkCutEE_ = conf.getParameter<int>("LoosePhotonHollowNTrkEE");
  loosephotonEtaWidthCutEE_ = conf.getParameter<double>("LoosePhotonEtaWidthEE");
  loosephotonHadOverEMCutEE_ = conf.getParameter<double>("LoosePhotonHadOverEMEE");
  loosephotonR9CutEE_ = conf.getParameter<double>("LoosePhotonR9CutEE");

  tightphotonEcalRecHitIsolationCutEE_ = conf.getParameter<double>("TightPhotonEcalRecHitIsoEE");
  tightphotonHcalRecHitIsolationCutEE_ = conf.getParameter<double>("TightPhotonHcalRecHitIsoEE");
  tightphotonHollowConeTrkIsolationCutEE_ = conf.getParameter<double>("TightPhotonHollowTrkEE");
  tightphotonSolidConeTrkIsolationCutEE_ = conf.getParameter<double>("TightPhotonSolidTrkEE");
  tightphotonSolidConeNTrkCutEE_ = conf.getParameter<int>("TightPhotonSolidNTrkEE");
  tightphotonHollowConeNTrkCutEE_ = conf.getParameter<int>("TightPhotonHollowNTrkEE");
  tightphotonEtaWidthCutEE_ = conf.getParameter<double>("TightPhotonEtaWidthEE");
  tightphotonHadOverEMCutEE_ = conf.getParameter<double>("TightPhotonHadOverEMEE");
  tightphotonR9CutEE_ = conf.getParameter<double>("TightPhotonR9CutEE");

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

//   std::cout << "Output from classification: " << std::endl;
//   std::cout << "Photon Eta: " << pho->p4().Eta();
//   std::cout << " Photon phi: " << pho->p4().Phi() << std::endl;
//   std::cout << "Flags: ";
//   std::cout << "isEBPho: " << isEBPho;
//   std::cout << " isEEPho: " << isEEPho;
//   std::cout << " isEBGap: " << isEBGap;
//   std::cout << " isEEGap: " << isEEGap;
//   std::cout << " isEBEEGap: " << isEBEEGap << std::endl;

  //Calculate hollow cone track isolation
  int ntrk=0;
  double trkiso=0;
  calculateTrackIso(pho, e, trkiso, ntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, trackConeInnerRadius_);

//   std::cout << "Output from hollow cone track isolation: ";
//   std::cout << " Sum pT: " << trkiso << " ntrk: " << ntrk << std::endl;

  //Calculate solid cone track isolation
  int sntrk=0;
  double strkiso=0;
  calculateTrackIso(pho, e, strkiso, sntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, 0.);

//   std::cout << "Output from solid cone track isolation: ";
//   std::cout << " Sum pT: " << strkiso << " ntrk: " << sntrk << std::endl;
  
  double EcalRecHitIso = calculateEcalRecHitIso(pho, e, es,
						photonEcalRecHitConeOuterRadius_,
						photonEcalRecHitConeInnerRadius_,
						photonEcalRecHitThresh_);
  double rawSCEt = (pho->superCluster()->rawEnergy())/(cosh(pho->superCluster()->position().eta()));
  double tempiso = EcalRecHitIso - rawSCEt;
  EcalRecHitIso= tempiso;

//   std::cout << "Output from ecal isolation: ";
//   std::cout << " Sum pT: " << EcalRecHitIso << std::endl;

  double HcalRecHitIso = calculateHcalRecHitIso(pho, e, es,
						photonHcalRecHitConeOuterRadius_,
						photonHcalRecHitConeInnerRadius_,
						photonHcalRecHitThresh_);

//   std::cout << "Output from hcal isolation: ";
//   std::cout << " Sum pT: " << HcalRecHitIso << std::endl;

  double EcalR9 = 0;
  //R9 calculation will go HERE.
  EcalR9 = calculateR9(pho, e, es);
  //

  bool isElec = isAlsoElectron(pho, e);

  //  std::cout << "Are you also an electron? " << isElec << std::endl;

  reco::PhotonID temp(false, false, false, strkiso,
		      trkiso, sntrk, ntrk,
		      EcalRecHitIso, HcalRecHitIso, EcalR9,
		      isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap,
		      isElec);
  if (isEBPho)
    decideEB(temp, pho);
  else
    decideEE(temp, pho);
  
  //  std::cout << "Cut based decision: " << temp.isLooseEM() << " " << temp.isLoosePhoton() <<  " " << temp.isTightPhoton() << std::endl;
  
  return temp;

}
void CutBasedPhotonIDAlgo::decideEB(reco::PhotonID &phID, const reco::Photon* pho){


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
    if(phID.isolationEcalRecHit() > looseEMEcalRecHitIsolationCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > looseEMHcalRecHitIsolationCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > looseEMSolidConeNTrkCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > looseEMHollowConeNTrkCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > looseEMSolidConeTrkIsolationCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > looseEMHollowConeTrkIsolationCutEB_){
      phID.setDecision(false, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (sigmaee > looseEMEtaWidthCutEB_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < looseEMR9CutEB_){
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
    if(phID.isolationEcalRecHit() > loosephotonEcalRecHitIsolationCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > loosephotonHcalRecHitIsolationCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > loosephotonSolidConeNTrkCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > loosephotonHollowConeNTrkCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > loosephotonSolidConeTrkIsolationCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > loosephotonHollowConeTrkIsolationCutEB_){
      phID.setDecision(true, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (sigmaee > loosephotonEtaWidthCutEB_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < loosephotonR9CutEB_){
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
    if(phID.isolationEcalRecHit() > tightphotonEcalRecHitIsolationCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > tightphotonHcalRecHitIsolationCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > tightphotonSolidConeNTrkCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > tightphotonHollowConeNTrkCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > tightphotonSolidConeTrkIsolationCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > tightphotonHollowConeTrkIsolationCutEB_){
      phID.setDecision(true, true, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    if (sigmaee > tightphotonEtaWidthCutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < tightphotonR9CutEB_){
      phID.setDecision(true, true, false);
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  phID.setDecision(true, true, true);
  
}



void CutBasedPhotonIDAlgo::decideEE(reco::PhotonID &phID, const reco::Photon* pho){


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
    if(phID.isolationEcalRecHit() > looseEMEcalRecHitIsolationCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > looseEMHcalRecHitIsolationCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > looseEMSolidConeNTrkCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > looseEMHollowConeNTrkCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > looseEMSolidConeTrkIsolationCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > looseEMHollowConeTrkIsolationCutEE_){
      phID.setDecision(false, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > looseEMHadOverEMCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    
    sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    
    if (sigmaee > looseEMEtaWidthCutEE_){
      phID.setDecision(false, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < looseEMR9CutEE_){
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
    if(phID.isolationEcalRecHit() > loosephotonEcalRecHitIsolationCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > loosephotonHcalRecHitIsolationCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > loosephotonSolidConeNTrkCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > loosephotonHollowConeNTrkCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > loosephotonSolidConeTrkIsolationCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > loosephotonHollowConeTrkIsolationCutEE_){
      phID.setDecision(true, false, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > loosephotonHadOverEMCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
  
    sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    
    if (sigmaee > loosephotonEtaWidthCutEE_){
      phID.setDecision(true, false, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < loosephotonR9CutEE_){
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
    if(phID.isolationEcalRecHit() > tightphotonEcalRecHitIsolationCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of hcal rec hits in a cone (HBHE)
  if(dophotonHcalRecHitIsolationCut_){
    if(phID.isolationHcalRecHit() > tightphotonHcalRecHitIsolationCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > tightphotonSolidConeNTrkCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > tightphotonHollowConeNTrkCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > tightphotonSolidConeTrkIsolationCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > tightphotonHollowConeTrkIsolationCutEE_){
      phID.setDecision(true, true, false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > tightphotonHadOverEMCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }

  //eta width

  if (dophotonsigmaeeCut_){
    reco::SuperClusterRef sc = pho->superCluster();
    
    double sigmaee = sc->etaWidth();
    
    sigmaee = sigmaee - 0.02*(fabs(sc->position().eta()) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
    
    if (sigmaee > tightphotonEtaWidthCutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }
  //R9 cut
  if (dophotonR9Cut_){
    if (phID.r9() < tightphotonR9CutEE_){
      phID.setDecision(true, true, false);
      return;
    }
  }


  //if you got here, you must have passed all cuts!
  phID.setDecision(true, true, true);
  
}
