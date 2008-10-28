#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * Reco Candidates with an Photon component
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Photon.h,v 1.21 2008/04/22 19:14:00 nancy Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// default constructor
    Photon() : RecoCandidate() { }
    /// constructor from values
    Photon( const LorentzVector & p4, Point caloPos, 
	    const SuperClusterRef scl, 
            float HoE,
	    bool hasPixelSeed=false, const Point & vtx = Point( 0, 0, 0 ) );
    /// destructor
    virtual ~Photon();
    /// returns a clone of the candidate
    virtual Photon * clone() const;
    /// reference to SuperCluster
    virtual reco::SuperClusterRef superCluster() const;
    /// vector of references to  Conversion's
    std::vector<reco::ConversionRef> conversions() const ; 
    /// set reference to SuperCluster
    void setSuperCluster( const reco::SuperClusterRef & r ) { superCluster_ = r; }
    /// add  single ConversionRef to the vector of Refs
    void addConversion( const reco::ConversionRef & r ) { conversions_.push_back(r); }
    /// set primary event vertex used to define photon direction
    void setVertex(const Point & vertex);
    /// set flags for photons in the ECAL fiducial volume
    void setFiducialVolumeFlags( const bool& a, 
				 const bool& b, 
				 const bool& c, 
				 const bool& d, 
				 const bool& e) 
      {  isEB_=a; isEE_=b; isEBGap_=c; isEEGap_=d; isEBEEGap_=e; } 
    /// set relevant shower shape variables 
    void setShowerShapeVariables ( const float& a, 
				   const float& b, 
				   const float& c, 
				   const float& d, 
				   const float& e, 
				   const float& f) 
      { e1x5_=a; e2x5_=b; e3x3_=c; e5x5_=d; covEtaEta_=e; covIetaIeta_=f;}
    /// set relevant isolation variables
    void setIsolationVariablesCone04 ( const float a, 
				 const float b, 
				 const float c, 
				 const float d, 
				 const int e, 
				 const int f  ) { 
      isolationEcalRecHitSum04_=a; 
      isolationHcalTowerSum04_=b;
      isolationSolidTrkCone04_=c;
      isolationHollowTrkCone04_=d;
      nTrkSolidCone04_=e;
      nTrkHollowCone04_=f;
    }
    void setIsolationVariablesCone03 ( const float a, 
				       const float b, 
				       const float c, 
				       const float d, 
				       const int e, 
				       const int f  ) { 
      isolationEcalRecHitSum03_=a; 
      isolationHcalTowerSum03_ =b;
      isolationSolidTrkCone03_ =c;
      isolationHollowTrkCone03_=d;
      nTrkSolidCone03_ =e;
      nTrkHollowCone03_=f;
    }


 
    /// set ID variables and output
    void setCutBasedIDOutput ( const bool a, const bool b, const bool c ) { cutBasedLooseEM_=a; cutBasedLoosePhoton_ =b;cutBasedTightPhoton_ =c;  } 
    ////////////  Retrieve quantities
    /// position in ECAL: this is th SC position if r9<0.93. If r8>0.93 is position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint caloPosition() const {return caloPosition_;}
    /// the hadronic over electromagnetic fraction
    float hadronicOverEm() const {return hadOverEm_;}
    /// Whether or not the SuperCluster has a matched GsfElectron pixel seed 
    bool hasPixelSeed() const { return pixelSeed_; }
    /// Bool flagging photons with a vector of refereces to conversions with size >0
    bool hasConversionTracks() const;
    /// Fiducial volume
    /// true if photon is in ECAL barrel
    bool isEB() const{return isEB_;}
    // true if photon is in ECAL endcap
    bool isEE() const{return isEE_;}
    /// true if photon is in EB, and inside the boundaries in super crystals/modules
    bool isEBGap() const{return isEBGap_;}
    /// true if photon is in EE, and inside the boundaries in supercrystal/D
    bool isEEGap() const{return isEEGap_;}
    /// true if photon is in boundary between EB and EE
    bool isEBEEGap() const{return isEBEEGap_;}
    ///  Shower shape variables
    float e1x5()         const {return e1x5_;}
    float e2x5()         const {return e2x5_;}
    float e3x3()         const {return e3x3_;}
    float e5x5()         const {return e5x5_;}
    float covEtaEta()    const {return covEtaEta_;}
    float covIetaIeta()  const {return covIetaIeta_;}
    float r1 ()          const {return e1x5_/e5x5_;}
    float r2 ()          const {return e2x5_/e5x5_;}
    float r9 ()          const {return e3x3_/this->superCluster()->rawEnergy();}  

    /// Isolation variables in cone dR=0.4
    ///Ecal isolation sum calculated from recHits
    float ecalRecHitSumDR04()      const{return isolationEcalRecHitSum04_;}
    /// Hcal isolation sum
    float hcalRecHitSumDR04()      const{return isolationHcalTowerSum04_;}
    //  Track pT sum c
    float isolationSolidTrkConeDR04()    const{return  isolationSolidTrkCone04_;}
    //As above, excluding the core at the center of the cone
    float isolationHollowTrkConeDR04()   const{return  isolationHollowTrkCone04_;}
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR04()              const{return  nTrkSolidCone04_;}
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR04()             const{return  nTrkHollowCone04_;}
    /// Isolation variables in cone dR=0.3
    float ecalRecHitSumDR03()      const{return isolationEcalRecHitSum03_;}
    /// Hcal isolation sum
    float hcalRecHitSumDR03()      const{return isolationHcalTowerSum03_;}
    //  Track pT sum c
    float isolationSolidTrkConeDR03()    const{return  isolationSolidTrkCone03_;}
    //As above, excluding the core at the center of the cone
    float isolationHollowTrkConeDR03()   const{return  isolationHollowTrkCone03_;}
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR03()              const{return  nTrkSolidCone03_;}
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR03()             const{return  nTrkHollowCone03_;}
    /// Cut based ID outputs
    bool isCutBasedLooseEM()     const{return cutBasedLooseEM_;}
    bool isCutBasedLoosePhoton() const{return cutBasedLoosePhoton_;}
    bool isCutBasedTightPhoton() const{return cutBasedTightPhoton_;}


  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPoint caloPosition_;
    /// reference to a SuperCluster
    reco::SuperClusterRef superCluster_;
    // vector of references to Conversions
    std::vector<reco::ConversionRef>  conversions_;

    float hadOverEm_;
    bool pixelSeed_;
    bool isEB_;
    bool isEE_;
    bool isEBGap_;
    bool isEEGap_;
    bool isEBEEGap_;
    /// shower shape variables
    float e1x5_;
    float e2x5_;
    float e3x3_;
    float e5x5_;
    float covEtaEta_;
    float covIetaIeta_;
    /// Isolation variables in cone dR=0.4
    float  isolationEcalRecHitSum04_;
    float  isolationHcalTowerSum04_;
    float  isolationSolidTrkCone04_;
    float  isolationHollowTrkCone04_;
    int  nTrkSolidCone04_;
    int  nTrkHollowCone04_;
    /// Isolation variables in cone dR=0.3
    float  isolationEcalRecHitSum03_;
    float  isolationHcalTowerSum03_;
    float  isolationSolidTrkCone03_;
    float  isolationHollowTrkCone03_;
    int  nTrkSolidCone03_;
    int  nTrkHollowCone03_;
    /// cut Based ID outputs
    bool cutBasedLooseEM_;
    bool cutBasedLoosePhoton_;
    bool cutBasedTightPhoton_;


  };
  
}

#endif
