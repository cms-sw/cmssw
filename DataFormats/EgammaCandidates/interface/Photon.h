#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * \author  N. Marinelli Univ. of Notre Dame
 * Photon object built out of PhotonCore
 * stores isolation, shower shape and additional info
 * needed for identification
 * 
 * \version $Id: Photon.h,v 1.33 2010/11/17 17:13:11 dlange Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// Forward declaration of data structures included in the object
    struct  FiducialFlags;
    struct  IsolationVariables;
    struct  ShowerShape;
    
    /// default constructor
    Photon() : RecoCandidate() { pixelSeed_=false; }

    /// copy constructor
    Photon ( const Photon&); 

    /// constructor from values
    Photon( const LorentzVector & p4, 
	    Point caloPos, 
	    const PhotonCoreRef & core,  
	    const Point & vtx = Point( 0, 0, 0 ) );

    /// destructor
    virtual ~Photon();

    /// returns a clone of the candidate
    virtual Photon * clone() const;

    /// returns a reference to the core photon object
    reco::PhotonCoreRef photonCore() const { return photonCore_;}
    //
    /// Retrieve photonCore attributes
    //
    /// Ref to SuperCluster
    reco::SuperClusterRef superCluster() const {return this->photonCore()->superCluster();}
    /// Ref to PFlow SuperCluster
    reco::SuperClusterRef pfSuperCluster() const {return this->photonCore()->pfSuperCluster();}
    /// Ref to PFlow canidate
    reco::PFCandidateRef pfCandidate() const {return this->photonCore()->pfCandidate();}
    /// vector of references to  Conversion's
    reco::ConversionRefVector conversions() const {return this->photonCore()->conversions() ;}  
    /// Bool flagging photons with a vector of refereces to conversions with size >0
    bool hasConversionTracks() const { if (this->photonCore()->conversions().size() > 0)  return true; else return false;}
    /// reference to electron Pixel seed 
    reco::ElectronSeedRefVector electronPixelSeeds() const {return this->photonCore()->electronPixelSeeds();}
    /// Bool flagging photons having a non-zero size vector of Ref to electornPixel seeds
    bool hasPixelSeed() const { if ((this->photonCore()->electronPixelSeeds()).size() > 0 ) return true; else return false; }


 
    /// position in ECAL: this is th SC position if r9<0.93. If r8>0.93 is position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint caloPosition() const {return caloPosition_;}
    /// set primary event vertex used to define photon direction
    void setVertex(const Point & vertex);
    /// Implement Candidate method for particle species
    bool isPhoton() const { return true ; }

    //=======================================================
    // Fiducial Flags
    //=======================================================
    struct  FiducialFlags
    {
      
      //Fiducial flags
      bool isEB;//Photon is in EB
      bool isEE;//Photon is in EE
      bool isEBEtaGap;//Photon is in supermodule/supercrystal eta gap in EB
      bool isEBPhiGap;//Photon is in supermodule/supercrystal phi gap in EB
      bool isEERingGap;//Photon is in crystal ring gap in EE
      bool isEEDeeGap;//Photon is in crystal dee gap in EE
      bool isEBEEGap;//Photon is in border between EB and EE.
      
      FiducialFlags():
        isEB(false),
           isEE(false),
           isEBEtaGap(false),
           isEBPhiGap(false),
           isEERingGap(false),
           isEEDeeGap(false),
           isEBEEGap(false)
           
      {}
      
      
    };

    /// set flags for photons in the ECAL fiducial volume
    void setFiducialVolumeFlags  ( const FiducialFlags&  a )  { fiducialFlagBlock_= a ;}
    /// Ritrievs fiducial flags
    /// true if photon is in ECAL barrel
    bool isEB() const{return  fiducialFlagBlock_.isEB;}
    // true if photon is in ECAL endcap
    bool isEE() const{return fiducialFlagBlock_.isEE;}
    /// true if photon is in EB, and inside the boundaries in super crystals/modules
    bool isEBGap() const { return (isEBEtaGap() || isEBPhiGap()); }
    bool isEBEtaGap() const{return fiducialFlagBlock_.isEBEtaGap;}
    bool isEBPhiGap() const{return fiducialFlagBlock_.isEBPhiGap;}
    /// true if photon is in EE, and inside the boundaries in supercrystal/D
    bool isEEGap() const { return (isEERingGap() || isEEDeeGap()); }
    bool isEERingGap() const{return fiducialFlagBlock_.isEERingGap;}
    bool isEEDeeGap() const{return fiducialFlagBlock_.isEEDeeGap;}
    /// true if photon is in boundary between EB and EE
    bool isEBEEGap() const{return fiducialFlagBlock_.isEBEEGap;}

    //=======================================================
    // Shower Shape Variables
    //=======================================================

    struct ShowerShape
    {
      float sigmaEtaEta ;
      float sigmaIetaIeta ;
      float e1x5 ;
      float e2x5 ;
      float e3x3 ;
      float e5x5 ;
      float maxEnergyXtal ; 
      float hcalDepth1OverEcal ; // hcal over ecal energy using first hcal depth
      float hcalDepth2OverEcal ; // hcal over ecal energy using 2nd hcal depth
      ShowerShape()
	: sigmaEtaEta(std::numeric_limits<float>::infinity()),
	   sigmaIetaIeta(std::numeric_limits<float>::infinity()),
	   e1x5(0), 
	   e2x5(0), 
	   e3x3(0), 
	   e5x5(0), 
	   maxEnergyXtal(0),
	   hcalDepth1OverEcal(0),
	   hcalDepth2OverEcal(0)
	   
      {}
    } ;
    void setShowerShapeVariables ( const ShowerShape& a )     { showerShapeBlock_ = a ;}
    /// the total hadronic over electromagnetic fraction
    float hadronicOverEm() const {return   showerShapeBlock_.hcalDepth1OverEcal + showerShapeBlock_.hcalDepth2OverEcal  ;}
    /// the  hadronic release in depth1 over electromagnetic fraction
    float hadronicDepth1OverEm() const {return  showerShapeBlock_.hcalDepth1OverEcal  ;}
    /// the  hadronic release in depth2 over electromagnetic fraction
    float hadronicDepth2OverEm() const {return  showerShapeBlock_.hcalDepth2OverEcal  ;}

    ///  Shower shape variables
    float e1x5()            const {return showerShapeBlock_.e1x5;}
    float e2x5()            const {return showerShapeBlock_.e2x5;}
    float e3x3()            const {return showerShapeBlock_.e3x3;}
    float e5x5()            const {return showerShapeBlock_.e5x5;}
    float maxEnergyXtal()   const {return showerShapeBlock_.maxEnergyXtal;}
    float sigmaEtaEta()     const {return showerShapeBlock_.sigmaEtaEta;}
    float sigmaIetaIeta()   const {return showerShapeBlock_.sigmaIetaIeta;}
    float r1x5 ()           const {return showerShapeBlock_.e1x5/showerShapeBlock_.e5x5;}
    float r2x5 ()           const {return showerShapeBlock_.e2x5/showerShapeBlock_.e5x5;}
    float r9 ()             const {return showerShapeBlock_.e3x3/this->superCluster()->rawEnergy();}  



    //=======================================================
    // Isolation Variables
    //=======================================================

    struct IsolationVariables
    {
      //These are analysis quantities calculated in the PhotonIDAlgo class
      
      //EcalRecHit isolation
      float ecalRecHitSumEt;
      //HcalDepth1Tower isolation
      float hcalTowerSumEt;
      //HcalDepth1Tower isolation
      float hcalDepth1TowerSumEt;
      //HcalDepth2Tower isolation
      float hcalDepth2TowerSumEt;
      //Sum of track pT in a cone of dR
      float trkSumPtSolidCone;
      //Sum of track pT in a hollow cone of outer radius, inner radius
      float trkSumPtHollowCone;
      //Number of tracks in a cone of dR
      int nTrkSolidCone;
      //Number of tracks in a hollow cone of outer radius, inner radius
      int nTrkHollowCone;
      
      IsolationVariables():
	
	ecalRecHitSumEt(0),
	   hcalTowerSumEt(0),
	   hcalDepth1TowerSumEt(0),
	   hcalDepth2TowerSumEt(0),
	   trkSumPtSolidCone(0),
	   trkSumPtHollowCone(0),
	   nTrkSolidCone(0),
	   nTrkHollowCone(0)
	   
      {}
      
      
    };


    struct PflowIsolationVariables
    {
            
      //EcalRecHit isolation
      float ecalRecHitSumEt;
      //HcalDepth1Tower isolation
      float hcalTowerSumEt;
      //HcalDepth1Tower isolation
      float hcalDepth1TowerSumEt;
      //HcalDepth2Tower isolation
      float hcalDepth2TowerSumEt;
      //Sum of track pT in a cone of dR
      float trkSumPtSolidCone;
      //Sum of track pT in a hollow cone of outer radius, inner radius
      float trkSumPtHollowCone;
      //Number of tracks in a cone of dR
      int nTrkSolidCone;
      //Number of tracks in a hollow cone of outer radius, inner radius
      int nTrkHollowCone;
      
      PflowIsolationVariables():
	
	ecalRecHitSumEt(0),
	hcalTowerSumEt(0),
	hcalDepth1TowerSumEt(0),
	hcalDepth2TowerSumEt(0),
	trkSumPtSolidCone(0),
	trkSumPtHollowCone(0),
	nTrkSolidCone(0),
	nTrkHollowCone(0)
	   
      {}
      
      
    };

    
    /// set relevant isolation variables
    void setIsolationVariables ( const IsolationVariables& isolInDr04, const IsolationVariables& isolInDr03) {  isolationR04_ = isolInDr04 ; isolationR03_ = isolInDr03 ;} 
    /// set isolation variables calculated with Pflow
    void setPflowIsolationVariables ( const PflowIsolationVariables& isolInDr04, const PflowIsolationVariables& isolInDr03) {  pfIsolationR04_ = isolInDr04 ; pfIsolationR03_ = isolInDr03 ;} 

    /// Egamma Isolation variables in cone dR=0.4
    ///Ecal isolation sum calculated from recHits
    float ecalRecHitSumEtConeDR04()      const{return  isolationR04_.ecalRecHitSumEt;}
    /// Hcal isolation sum
    float hcalTowerSumEtConeDR04()      const{return  isolationR04_.hcalTowerSumEt ;}
    /// Hcal-Depth1 isolation sum
    float hcalDepth1TowerSumEtConeDR04()      const{return  isolationR04_.hcalDepth1TowerSumEt;}
    /// Hcal-Depth2 isolation sum
    float hcalDepth2TowerSumEtConeDR04()      const{return  isolationR04_.hcalDepth2TowerSumEt;}
    //  Track pT sum c
    float trkSumPtSolidConeDR04()    const{return   isolationR04_.trkSumPtSolidCone;}
    //As above, excluding the core at the center of the cone
    float trkSumPtHollowConeDR04()   const{return   isolationR04_.trkSumPtHollowCone;}
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR04()              const{return   isolationR04_.nTrkSolidCone;}
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR04()            const{return   isolationR04_.nTrkHollowCone;}
    //
    /// Isolation variables in cone dR=0.3
    float ecalRecHitSumEtConeDR03()      const{return  isolationR03_.ecalRecHitSumEt;}
    /// Hcal isolation sum
    float hcalTowerSumEtConeDR03()      const{return isolationR03_.hcalTowerSumEt;}
    /// Hcal-Depth1 isolation sum
    float hcalDepth1TowerSumEtConeDR03()      const{return isolationR03_.hcalDepth1TowerSumEt;}
    /// Hcal-Depth2 isolation sum
    float hcalDepth2TowerSumEtConeDR03()      const{return isolationR03_.hcalDepth2TowerSumEt;}
    //  Track pT sum c
    float trkSumPtSolidConeDR03()    const{return  isolationR03_.trkSumPtSolidCone;}
    //As above, excluding the core at the center of the cone
    float trkSumPtHollowConeDR03()   const{return  isolationR03_.trkSumPtHollowCone;}
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR03()              const{return  isolationR03_.nTrkSolidCone;}
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR03()             const{return  isolationR03_.nTrkHollowCone;}



    /// Particle Flow Isolation variables in cone dR=0.4
    ///Ecal isolation sum calculated from recHits
    float PFecalRecHitSumEtConeDR04()      const{return  pfIsolationR04_.ecalRecHitSumEt;}
    /// Hcal isolation sum
    float PFhcalTowerSumEtConeDR04()      const{return  pfIsolationR04_.hcalTowerSumEt ;}
    /// Hcal-Depth1 isolation sum
    float PFhcalDepth1TowerSumEtConeDR04()      const{return  pfIsolationR04_.hcalDepth1TowerSumEt;}
    /// Hcal-Depth2 isolation sum
    float PFhcalDepth2TowerSumEtConeDR04()      const{return  pfIsolationR04_.hcalDepth2TowerSumEt;}
    //  Track pT sum c
    float PFtrkSumPtSolidConeDR04()    const{return   pfIsolationR04_.trkSumPtSolidCone;}
    //As above, excluding the core at the center of the cone
    float PFtrkSumPtHollowConeDR04()   const{return   pfIsolationR04_.trkSumPtHollowCone;}
    //Returns number of tracks in a cone of dR
    int PFnTrkSolidConeDR04()              const{return   pfIsolationR04_.nTrkSolidCone;}
    //As above, excluding the core at the center of the cone
    int PFnTrkHollowConeDR04()            const{return   pfIsolationR04_.nTrkHollowCone;}
    //
    /// Particle Flow Isolation variables in cone dR=0.3
    float PFecalRecHitSumEtConeDR03()      const{return  pfIsolationR03_.ecalRecHitSumEt;}
    /// Hcal isolation sum
    float PFhcalTowerSumEtConeDR03()      const{return pfIsolationR03_.hcalTowerSumEt;}
    /// Hcal-Depth1 isolation sum
    float PFhcalDepth1TowerSumEtConeDR03()      const{return pfIsolationR03_.hcalDepth1TowerSumEt;}
    /// Hcal-Depth2 isolation sum
    float PFhcalDepth2TowerSumEtConeDR03()      const{return pfIsolationR03_.hcalDepth2TowerSumEt;}
    //  Track pT sum c
    float PFtrkSumPtSolidConeDR03()    const{return  pfIsolationR03_.trkSumPtSolidCone;}
    //As above, excluding the core at the center of the cone
    float PFtrkSumPtHollowConeDR03()   const{return  pfIsolationR03_.trkSumPtHollowCone;}
    //Returns number of tracks in a cone of dR
    int PFnTrkSolidConeDR03()              const{return  pfIsolationR03_.nTrkSolidCone;}
    //As above, excluding the core at the center of the cone
    int PFnTrkHollowConeDR03()             const{return  pfIsolationR03_.nTrkHollowCone;}





  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint caloPosition_;
    /// reference to the PhotonCore
    reco::PhotonCoreRef photonCore_;
    //
    bool pixelSeed_;
    //
    FiducialFlags fiducialFlagBlock_;
    IsolationVariables isolationR04_;
    IsolationVariables isolationR03_;
    ShowerShape        showerShapeBlock_;
    PflowIsolationVariables pfIsolationR04_;
    PflowIsolationVariables pfIsolationR03_;


   


  };
  
}

#endif
