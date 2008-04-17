#ifndef PhotonID_h
#define PhotonID_h
#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
namespace reco {

class PhotonID
{
 public:

  PhotonID(); //Default constructor.  It avails ye naught.

  //Real constructor
  PhotonID(bool Decision, double BCIso, double TrkCone4,
	   double HollowCone4, int nTrkCone4, int nHollow4,
	   bool EBPho, bool EEPho, bool EBGap, bool EEGap, bool EBEEGap,
	   bool isAlsoElectron);
  

  //getters:
  
  //Returns decision based on the cuts in the configuration file in Algo
  bool cutBasedDecision() const {return cutBasedDecision_;}
  //Returns computed BasicCluster isolation
  double isolationECal() const{return isolationECal_;}
  //Returns calculated sum track pT cone of dR
  double isolationSolidTrkCone() const{return isolationSolidTrkCone_;}
  //As above, excluding the core at the center of the cone
  double isolationHollowTrkCone() const{return isolationHollowTrkCone_;}
  //Returns number of tracks in a cone of dR
  int nTrkSolidCone() const{return nTrkSolidCone_;}
  //As above, excluding the core at the center of the cone
  int nTrkHollowCone() const{return nTrkHollowTrkCone_;}

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
  void setDecision(bool decision);

 private:

  //Did this pass the cuts in the configuration?
  bool cutBasedDecision_;

  //These are analysis quantities calculated in the algo class

  //BasicCluster Isolation
  double isolationECal_;
  //Sum of track pT in a cone of dR=0.4
  double isolationSolidTrkCone_;
  //Sum of track pT in a hollow cone of outer radius 0.4, inner radius 0.1
  double isolationHollowTrkCone_;
  //Number of tracks in a cone of dR=0.4
  int nTrkSolidCone_;
  //Number of tracks in a hollow cone of outer radius 0.4, inner radius 0.1
  int nTrkHollowTrkCone_;
 
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
