#include "RecoEgamma/PhotonIdentification/interface/CutBasedPhotonIDAlgo.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

void CutBasedPhotonIDAlgo::setup(const edm::ParameterSet& conf) {
  
  // Get all the parameters
  baseSetup(conf);

  //get cuts here
  photonBasicClusterIsolationCut_ = conf.getParameter<double>("PhotonBCIso");
  photonHollowConeTrkIsolationCut_ = conf.getParameter<double>("PhotonHollowTrk");
  photonSolidConeTrkIsolationCut_ = conf.getParameter<double>("PhotonSolidTrk");
  photonSolidConeNTrkCut_ = conf.getParameter<int>("PhotonSolidNTrk");
  photonHollowConeNTrkCut_ = conf.getParameter<int>("PhotonHollowNTrk");
  photonEtaWidthCut_ = conf.getParameter<double>("PhotonEtaWidth");
  photonHadOverEMCut_ = conf.getParameter<double>("PhotonHadOverEM");

  photonBasicClusterConeOuterRadius_ = conf.getParameter<double>("BasicClusterConeOuterRadius");
  photonBasicClusterConeInnerRadius_ = conf.getParameter<double>("BasicClusterConeInnerRadius");
  isolationbasicclusterThreshold_ = conf.getParameter<double>("isolationbasicclusterThreshold");
  trackConeOuterRadius_ = conf.getParameter<double>("TrackConeOuterRadius");
  trackConeInnerRadius_ = conf.getParameter<double>("TrackConeInnerRadius");
  isolationtrackThreshold_ = conf.getParameter<double>("isolationtrackThreshold");

  //Decision cuts
  dophotonBCIsolationCut_ = conf.getParameter<bool>("DoBasicClusterIsolationCut");
  dophotonHCTrkIsolationCut_ = conf.getParameter<bool>("DoHollowConeTrackIsolationCut");
  dophotonSCTrkIsolationCut_ = conf.getParameter<bool>("DoSolidConeTrackIsolationCut");
  dophotonHCNTrkCut_ = conf.getParameter<bool>("DoHollowConeNTrkCut");
  dophotonSCNTrkCut_ = conf.getParameter<bool>("DoSolidConeNTrkCut");
  dophotonHadOverEMCut_ = conf.getParameter<bool>("DoHadOverEMCut");
  dophotonsigmaeeCut_ = conf.getParameter<bool>("DoEtaWidthCut");
  dorequireNotElectron_ = conf.getParameter<bool>("RequireNotElectron");
  dorequireFiducial_ = conf.getParameter<bool>("RequireFiducial");
}

reco::PhotonID CutBasedPhotonIDAlgo::calculate(const reco::Photon* pho, const edm::Event& e){

  //need to do the following things here:
  //1.)  Call base class methods to calculate photonID variables like fiducial and
  //     isolations.
  //2.)  Decide whether this particular photon passes the cuts that are set forth in the ps.
  //3.)  Create a new PhotonID object, complete with decision and return it.


  //Get fiducial information
  bool isEBPho   = false;
  bool isEEPho   = false;
  bool isEBGap   = false;
  bool isEEGap   = false;
  bool isEBEEGap = false;
  classify(pho, isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap);

  //Calculate hollow cone track isolation
  int ntrk=0;
  double trkiso=0;
  calculateTrackIso(pho, e, trkiso, ntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, trackConeInnerRadius_);

  //Calculate solid cone track isolation
  int sntrk=0;
  double strkiso=0;
  calculateTrackIso(pho, e, strkiso, sntrk, isolationtrackThreshold_,    
		    trackConeOuterRadius_, 0.);
  
  double bc_iso = calculateBasicClusterIso(pho, e, 
					   photonBasicClusterConeOuterRadius_,
					   photonBasicClusterConeInnerRadius_,
					   isolationbasicclusterThreshold_);

  bool isElec = isAlsoElectron(pho, e);
  
  reco::PhotonID temp(false, bc_iso, strkiso,
		      trkiso, sntrk, ntrk,
		      isEBPho, isEEPho, isEBGap, isEEGap, isEBEEGap,
		      isElec);

  decide(temp, pho);
  return temp;

}
void CutBasedPhotonIDAlgo::decide(reco::PhotonID &phID, const reco::Photon* pho){

  //Require that this is not also an Electron supercluster
  if (dorequireNotElectron_){
    if (phID.isAlsoElectron()){
      phID.setDecision(false);
      return;
    }
  }
  
  //Require supercluster is within fiducial volume.
  if(dorequireFiducial_){
    if (phID.isEBEEGap()) {
      phID.setDecision(false);
      return;
    }
    if (phID.isEBPho() && phID.isEBGap()){ 
      phID.setDecision(false);
      return;
    }
    if (phID.isEEPho() && phID.isEEGap()){
      phID.setDecision(false);
      return;
    }
  }
  
  //Cut on the sum of basic clusters within a cone
  if(dophotonBCIsolationCut_){
    if (phID.isolationHollowTrkCone() > photonBasicClusterIsolationCut_){
      phID.setDecision(false);
      return;
    }
  }

  //Cut on number of tracks within the solid cone.
  if (dophotonSCNTrkCut_){
    if (phID.nTrkSolidCone() > photonSolidConeNTrkCut_){
      phID.setDecision(false);
      return;
    }
  }

  //Cut on number of tracks within the hollow cone.
  if (dophotonHCNTrkCut_){
    if (phID.nTrkHollowCone() > photonHollowConeNTrkCut_){
      phID.setDecision(false);
      return;
    }
  }
  
  //Cut on the sum of tracks within a solid cone
  if (dophotonSCTrkIsolationCut_){
    if (phID.isolationSolidTrkCone() > photonSolidConeTrkIsolationCut_){
      phID.setDecision(false);
      return;
    }
  }

  //Cut on the sum of tracks within a hollow cone
  if (dophotonHCTrkIsolationCut_){
    if (phID.isolationHollowTrkCone() > photonHollowConeTrkIsolationCut_){
      phID.setDecision(false);
      return;
    }  
  }

  //HadoverEM cut
  if (dophotonHadOverEMCut_){
    float hadoverE = pho->hadronicOverEm();
    if (hadoverE > photonHadOverEMCut_){
      phID.setDecision(false);
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
    if (sigmaee > photonEtaWidthCut_){
      phID.setDecision(false);
      return;
    }
  }

  //if you got here, you must have passed all cuts!
  phID.setDecision(true);
  

}
