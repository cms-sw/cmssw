#ifndef  EgammaCandidates_CutBasedPhotonID_H
#define  EgammaCandidates_CutBasedPhotonID_H



  struct PhotonFiducialFlags
  {
    
    //Fiducial flags
    bool isEBPho;//Photon is in EB
    bool isEEPho;//Photon is in EE
    bool isEBGap;//Photon is in supermodule/supercrystal gap in EB
    bool isEEGap;//Photon is in crystal gap in EE
    bool isEBEEGap;//Photon is in border between EB and EE.
    
    PhotonFiducialFlags():
      isEBPho(false),
         isEEPho(false),
         isEBGap(false),
	 isEEGap(false),
	 isEBEEGap(false)
	 
    {}
    
    
  };



  struct PhotonIsolationVariables
  {
    //These are analysis quantities calculated in the PhotonIDAlgo class

    //EcalRecHit isolation
    float isolationEcalRecHit;
    //HcalTower isolation
    float isolationHcalTower;
    //HcalDepth1Tower isolation
    float isolationHcalDepth1Tower;
    //HcalDepth2Tower isolation
    float isolationHcalDepth2Tower;
    //Sum of track pT in a cone of dR
    float isolationSolidTrkCone;
    //Sum of track pT in a hollow cone of outer radius, inner radius
    float isolationHollowTrkCone;
    //Number of tracks in a cone of dR
    int nTrkSolidCone;
    //Number of tracks in a hollow cone of outer radius, inner radius
    int nTrkHollowCone;
    
    PhotonIsolationVariables():

	 isolationEcalRecHit(0),
	 isolationHcalTower(0),
	 isolationHcalDepth1Tower(0),
	 isolationHcalDepth2Tower(0),
	 isolationSolidTrkCone(0),
	 isolationHollowTrkCone(0),
         nTrkSolidCone(0),
         nTrkHollowCone(0)
	 
    {}
    
    
  };
  

  struct CutBasedPhotonID
  {
    //Did this pass the cuts in the configuration?
    bool isLooseEM;
    bool isLoosePhoton;
    bool isTightPhoton;
    
    
    CutBasedPhotonID():
         isLooseEM(false),
	 isLoosePhoton(false),
	 isTightPhoton(false)
	 
    {}
    
    
  };
  






#endif
