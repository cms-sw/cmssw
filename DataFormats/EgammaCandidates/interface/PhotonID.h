#ifndef PhotonID_h
#define PhotonID_h
#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
namespace reco {

class PhotonID
{
 public:

  PhotonID(); //Default constructor.  It avails ye naught.

  //Real constructor
  PhotonID(bool LooseEM, bool LoosePho, bool TightPho, 
	   float TrkCone4, float HollowCone, int nTrkCone, int nHollow,
	   float EcalRecHitIso, float HcalRecHitIso, float r9,
	   bool EBPho, bool EEPho, bool EBGap, bool EEGap, bool EBEEGap,
	   bool isAlsoElectron);
  

  //getters:
  
  //Returns decision based on the cuts in the configuration file in 
  //RecoEgamma/PhotonIdentification
  bool isLooseEM() const{return isLooseEM_;}
  bool isLoosePhoton() const{return isLoosePhoton_;}
  bool isTightPhoton() const{return isTightPhoton_;}

  //Returns computed EcalRecHit isolation
  float isolationEcalRecHit() const{return isolationEcalRecHit_;}
  //Returns computed HcalRecHit isolation
  float isolationHcalRecHit() const{return isolationHcalRecHit_;}
  //Returns calculated sum track pT cone of dR
  float isolationSolidTrkCone() const{return isolationSolidTrkCone_;}
  //As above, excluding the core at the center of the cone
  float isolationHollowTrkCone() const{return isolationHollowTrkCone_;}
  //Returns number of tracks in a cone of dR
  int nTrkSolidCone() const{return nTrkSolidCone_;}
  //As above, excluding the core at the center of the cone
  int nTrkHollowCone() const{return nTrkHollowTrkCone_;}
  //return r9 = e3x3/etotal
  float r9() const{return r9_;}
  //if photon is in ECAL barrel
  bool isEBPho() const{return isEBPho_;}
  //if photon is in ECAL endcap
  bool isEEPho() const{return isEEPho_;}
  //if photon is in EB, and inside the boundaries in super crystals/modules
  bool isEBGap() const{return isEBGap_;}
  //if photon is in EE, and inside the boundaries in supercrystal/D
  bool isEEGap() const{return isEEGap_;}
  //if photon is in boundary between EB and EE
  bool isEBEEGap() const{return isEBEEGap_;}
  //if this is also a GsfElectron
  bool isAlsoElectron() const{return isAlsoElectron_;}

  //setters:
  void setFiducialFlags(bool EBPho, bool EEPho, bool EBGap, bool EEGap, bool EBEEGap);
  void setDecision(bool decisionLooseEM, bool decisionLoosePho, bool decisionTightPho);

 private:

  //Did this pass the cuts in the configuration?
  bool isLooseEM_;
  bool isLoosePhoton_;
  bool isTightPhoton_;

  //These are analysis quantities calculated in the PhotonIDAlgo class
  //EcalRecHit isolation
  float isolationEcalRecHit_;
  //HcalRecHit isolation
  float isolationHcalRecHit_;
  //Sum of track pT in a cone of dR
  float isolationSolidTrkCone_;
  //Sum of track pT in a hollow cone of outer radius, inner radius
  float isolationHollowTrkCone_;
  //Number of tracks in a cone of dR
  int nTrkSolidCone_;
  //Number of tracks in a hollow cone of outer radius, inner radius
  int nTrkHollowTrkCone_;
  //r9 variable
  float r9_;
  //Fiducial flags
  bool isEBPho_;//Photon is in EB
  bool isEEPho_;//Photon is in EE
  bool isEBGap_;//Photon is in supermodule/supercrystal gap in EB
  bool isEEGap_;//Photon is in crystal gap in EE
  bool isEBEEGap_;//Photon is in border between EB and EE.

  //Electron identity?
  bool isAlsoElectron_;
};

}

#endif
