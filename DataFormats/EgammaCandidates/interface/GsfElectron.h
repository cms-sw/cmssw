 #ifndef GsfElectron_h
#define GsfElectron_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
//#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <limits>

namespace reco
 {


/****************************************************************************
 * \class reco::GsfElectron
 *
 * An Electron with a GsfTrack seeded from an ElectronSeed.
 * Renamed from PixelMatchGsfElectron.
 * Originally adapted from the TRecElectron class in ORCA.
 *
 * \author Claude Charlot - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 * \author David Chamont  - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 ****************************************************************************/

class GsfElectron : public RecoCandidate
 {

  //=======================================================
  // Constructors
  //
  // The clone() method with arguments, and the copy
  // constructor with edm references is designed for
  // someone which want to duplicates all
  // collections.
  //=======================================================

  public :

    // some nested structures defined later on
    struct ChargeInfo ;
    struct TrackClusterMatching ;
    struct TrackExtrapolations ;
    struct ClosestCtfTrack ;
    struct FiducialFlags ;
    struct ShowerShape ;
    struct IsolationVariables ;
    struct ConversionRejection ;
    struct ClassificationVariables ;

    GsfElectron() ;
    GsfElectron( const GsfElectronCoreRef & ) ;
    GsfElectron
     (
      const GsfElectron &,
      const GsfElectronCoreRef &
     ) ;
    GsfElectron
     (
      const GsfElectron & electron,
      const GsfElectronCoreRef & core,
      const CaloClusterPtr & electronCluster,
      const TrackRef & closestCtfTrack,
      const TrackBaseRef & conversionPartner,
      const GsfTrackRefVector & ambiguousTracks
     ) ;
    GsfElectron
     (
      int charge,
      const ChargeInfo &,
      const GsfElectronCoreRef &,
      const TrackClusterMatching &,
      const TrackExtrapolations &,
      const ClosestCtfTrack &,
      const FiducialFlags &,
      const ShowerShape &,
      const ConversionRejection &
     ) ;
    GsfElectron * clone() const ;
    GsfElectron * clone
     (
      const GsfElectronCoreRef & core,
      const CaloClusterPtr & electronCluster,
      const TrackRef & closestCtfTrack,
      const TrackBaseRef & conversionPartner,
      const GsfTrackRefVector & ambiguousTracks
     ) const ;
    virtual ~GsfElectron() {} ;

  private:

    void init() ;


  //=======================================================
  // Candidate methods and complementary information
  //
  // The gsf electron producer has tried to best evaluate
  // the four momentum and charge and given those values to
  // the GsfElectron constructor, which forwarded them to
  // the Candidate constructor. Those values can be retreived
  // with getters inherited from Candidate : p4() and charge().
  //=======================================================

  public:

    // Inherited from Candidate
    // const LorentzVector & charge() const ;
    // const LorentzVector & p4() const ;

    // Complementary struct
    struct ChargeInfo
     {
      int scPixCharge ;
      bool isGsfCtfScPixConsistent ;
      bool isGsfScPixConsistent ;
      bool isGsfCtfConsistent ;
      ChargeInfo()
        : scPixCharge(0), isGsfCtfScPixConsistent(false),
          isGsfScPixConsistent(false), isGsfCtfConsistent(false)
        {}
     } ;

    // Charge info accessors
    // to get gsf track charge: gsfTrack()->charge()
    // to get ctf track charge, if closestCtfTrackRef().isNonnull(): closestCtfTrackRef()->charge()
    int scPixCharge() const { return chargeInfo_.scPixCharge ; }
    bool isGsfCtfScPixChargeConsistent() const { return chargeInfo_.isGsfCtfScPixConsistent ; }
    bool isGsfScPixChargeConsistent() const { return chargeInfo_.isGsfScPixConsistent ; }
    bool isGsfCtfChargeConsistent() const { return chargeInfo_.isGsfCtfConsistent ; }
    const ChargeInfo & chargeInfo() const { return chargeInfo_ ; }

    // Candidate redefined methods
    virtual bool isElectron() const { return true ; }
    virtual bool overlap( const Candidate & ) const ;

  private:

    // Complementary attributes
    ChargeInfo chargeInfo_ ;


  //=======================================================
  // Core Attributes
  //
  // They all have been computed before, when building the
  // collection of GsfElectronCore instances. Each GsfElectron
  // has a reference toward a GsfElectronCore.
  //=======================================================

  public:

    // accessors
    virtual GsfElectronCoreRef core() const ;

    // forward core methods
    virtual SuperClusterRef superCluster() const { return core()->superCluster() ; }
    virtual GsfTrackRef gsfTrack() const { return core()->gsfTrack() ; }
    virtual TrackRef closestTrack() const { return core()->ctfTrack() ; }
    float ctfGsfOverlap() const { return core()->ctfGsfOverlap() ; }
    bool ecalDrivenSeed() const { return core()->ecalDrivenSeed() ; }
    bool trackerDrivenSeed() const { return core()->trackerDrivenSeed() ; }
    SuperClusterRef parentSuperCluster() const { return core()->parentSuperCluster() ; }

    // backward compatibility
    struct ClosestCtfTrack
     {
      TrackRef ctfTrack ; // best matching ctf track
      float shFracInnerHits ; // fraction of common hits between the ctf and gsf tracks
      ClosestCtfTrack() : shFracInnerHits(0.) {}
      ClosestCtfTrack( TrackRef track, float sh ) : ctfTrack(track), shFracInnerHits(sh) {}
     } ;
    float shFracInnerHits() const { return core()->ctfGsfOverlap() ; }
    TrackRef closestCtfTrackRef() const { return core()->ctfTrack() ; }
    ClosestCtfTrack closestCtfTrack() const { return ClosestCtfTrack(core()->ctfTrack(),core()->ctfGsfOverlap()) ; }

  private:

    // attributes
    GsfElectronCoreRef core_ ;


  //=======================================================
  // Track-Cluster Matching Attributes
  //=======================================================

  public:

    struct TrackClusterMatching
     {
      CaloClusterPtr electronCluster ;  // basic cluster best matching gsf track
      float eSuperClusterOverP ;        // the supercluster energy / track momentum at the PCA to the beam spot
      float eSeedClusterOverP ;         // the seed cluster energy / track momentum at the PCA to the beam spot
      float eSeedClusterOverPout ;      // the seed cluster energy / track momentum at calo extrapolated from the outermost track state
      float eEleClusterOverPout ;       // the electron cluster energy / track momentum at calo extrapolated from the outermost track state
      float deltaEtaSuperClusterAtVtx ; // the supercluster eta - track eta position at calo extrapolated from innermost track state
      float deltaEtaSeedClusterAtCalo ; // the seed cluster eta - track eta position at calo extrapolated from the outermost track state
      float deltaEtaEleClusterAtCalo ;  // the electron cluster eta - track eta position at calo extrapolated from the outermost state
      float deltaPhiEleClusterAtCalo ;  // the electron cluster phi - track phi position at calo extrapolated from the outermost track state
      float deltaPhiSuperClusterAtVtx ; // the supercluster phi - track phi position at calo extrapolated from the innermost track state
      float deltaPhiSeedClusterAtCalo ; // the seed cluster phi - track phi position at calo extrapolated from the outermost track state
      TrackClusterMatching()
       : eSuperClusterOverP(0.),
         eSeedClusterOverP(0.),
         eSeedClusterOverPout(0.),
         eEleClusterOverPout(0.),
         deltaEtaSuperClusterAtVtx(std::numeric_limits<float>::max()),
         deltaEtaSeedClusterAtCalo(std::numeric_limits<float>::max()),
         deltaEtaEleClusterAtCalo(std::numeric_limits<float>::max()),
         deltaPhiEleClusterAtCalo(std::numeric_limits<float>::max()),
         deltaPhiSuperClusterAtVtx(std::numeric_limits<float>::max()),
         deltaPhiSeedClusterAtCalo(std::numeric_limits<float>::max())
        {}
     } ;

    // accessors
    CaloClusterPtr electronCluster() const { return trackClusterMatching_.electronCluster ; }
    float eSuperClusterOverP() const { return trackClusterMatching_.eSuperClusterOverP ; }
    float eSeedClusterOverP() const { return trackClusterMatching_.eSeedClusterOverP ; }
    float eSeedClusterOverPout() const { return trackClusterMatching_.eSeedClusterOverPout ; }
    float eEleClusterOverPout() const { return trackClusterMatching_.eEleClusterOverPout ; }
    float deltaEtaSuperClusterTrackAtVtx() const { return trackClusterMatching_.deltaEtaSuperClusterAtVtx ; }
    float deltaEtaSeedClusterTrackAtCalo() const { return trackClusterMatching_.deltaEtaSeedClusterAtCalo ; }
    float deltaEtaEleClusterTrackAtCalo() const { return trackClusterMatching_.deltaEtaEleClusterAtCalo ; }
    float deltaPhiSuperClusterTrackAtVtx() const { return trackClusterMatching_.deltaPhiSuperClusterAtVtx ; }
    float deltaPhiSeedClusterTrackAtCalo() const { return trackClusterMatching_.deltaPhiSeedClusterAtCalo ; }
    float deltaPhiEleClusterTrackAtCalo() const { return trackClusterMatching_.deltaPhiEleClusterAtCalo ; }
    const TrackClusterMatching & trackClusterMatching() const { return trackClusterMatching_ ; }

    // for backward compatibility, usefull ?
    void setDeltaEtaSuperClusterAtVtx( float de ) { trackClusterMatching_.deltaEtaSuperClusterAtVtx = de ; }
    void setDeltaPhiSuperClusterAtVtx( float dphi ) { trackClusterMatching_.deltaPhiSuperClusterAtVtx = dphi ; }


  private:

    // attributes
    TrackClusterMatching trackClusterMatching_ ;


  //=======================================================
  // Track extrapolations
  //=======================================================

  public :

    struct TrackExtrapolations
     {
      math::XYZPointF  positionAtVtx ;     // the track PCA to the beam spot
      math::XYZPointF  positionAtCalo ;    // the track PCA to the supercluster position
      math::XYZVectorF momentumAtVtx ;     // the track momentum at the PCA to the beam spot
      math::XYZVectorF momentumAtCalo ;    // the track momentum extrapolated at the supercluster position from the innermost track state
      math::XYZVectorF momentumOut ;       // the track momentum extrapolated at the seed cluster position from the outermost track state
      math::XYZVectorF momentumAtEleClus ; // the track momentum extrapolated at the ele cluster position from the outermost track state
      math::XYZVectorF momentumAtVtxWithConstraint ;     // the track momentum at the PCA to the beam spot using bs constraint
     } ;

    // accessors
    math::XYZPointF trackPositionAtVtx() const { return trackExtrapolations_.positionAtVtx ; }
    math::XYZPointF trackPositionAtCalo() const { return trackExtrapolations_.positionAtCalo ; }
    math::XYZVectorF trackMomentumAtVtx() const { return trackExtrapolations_.momentumAtVtx ; }
    math::XYZVectorF trackMomentumAtCalo() const { return trackExtrapolations_.momentumAtCalo ; }
    math::XYZVectorF trackMomentumOut() const { return trackExtrapolations_.momentumOut ; }
    math::XYZVectorF trackMomentumAtEleClus() const { return trackExtrapolations_.momentumAtEleClus ; }
    math::XYZVectorF trackMomentumAtVtxWithConstraint() const { return trackExtrapolations_.momentumAtVtxWithConstraint ; }
    const TrackExtrapolations & trackExtrapolations() const { return trackExtrapolations_ ; }

    // setter (if you know what you're doing)
    void setTrackExtrapolations(const TrackExtrapolations &te) { trackExtrapolations_ = te; }

    // for backward compatibility
    math::XYZPointF TrackPositionAtVtx() const { return trackPositionAtVtx() ; }
    math::XYZPointF TrackPositionAtCalo() const { return trackPositionAtCalo() ; }


  private:

    // attributes
    TrackExtrapolations trackExtrapolations_ ;


  //=======================================================
  // SuperCluster direct access
  //=======================================================

  public :

    // direct accessors
    math::XYZPoint superClusterPosition() const { return superCluster()->position() ; } // the super cluster position
    int basicClustersSize() const { return superCluster()->clustersSize() ; } // number of basic clusters inside the supercluster
    CaloCluster_iterator basicClustersBegin() const { return superCluster()->clustersBegin() ; }
    CaloCluster_iterator basicClustersEnd() const { return superCluster()->clustersEnd() ; }

    // for backward compatibility
    math::XYZPoint caloPosition() const { return superCluster()->position() ; }



  //=======================================================
  // Fiducial Flags
  //=======================================================

  public :

    struct FiducialFlags
     {
      bool isEB ;        // true if particle is in ECAL Barrel
      bool isEE ;        // true if particle is in ECAL Endcaps
      bool isEBEEGap ;   // true if particle is in the crack between EB and EE
      bool isEBEtaGap ;  // true if particle is in EB, and inside the eta gaps between modules
      bool isEBPhiGap ;  // true if particle is in EB, and inside the phi gaps between modules
      bool isEEDeeGap ;  // true if particle is in EE, and inside the gaps between dees
      bool isEERingGap ; // true if particle is in EE, and inside the gaps between rings
      FiducialFlags()
       : isEB(false), isEE(false), isEBEEGap(false),
         isEBEtaGap(false), isEBPhiGap(false),
	     isEEDeeGap(false), isEERingGap(false)
	   {}
     } ;

    // accessors
    bool isEB() const { return fiducialFlags_.isEB ; }
    bool isEE() const { return fiducialFlags_.isEE ; }
    bool isGap() const { return ((isEBEEGap())||(isEBGap())||(isEEGap())) ; }
    bool isEBEEGap() const { return fiducialFlags_.isEBEEGap ; }
    bool isEBGap() const { return (isEBEtaGap()||isEBPhiGap()) ; }
    bool isEBEtaGap() const { return fiducialFlags_.isEBEtaGap ; }
    bool isEBPhiGap() const { return fiducialFlags_.isEBPhiGap ; }
    bool isEEGap() const { return (isEEDeeGap()||isEERingGap()) ; }
    bool isEEDeeGap() const { return fiducialFlags_.isEEDeeGap ; }
    bool isEERingGap() const { return fiducialFlags_.isEERingGap ; }
    const FiducialFlags & fiducialFlags() const { return fiducialFlags_ ; }


  private:

    // attributes
    FiducialFlags fiducialFlags_ ;


  //=======================================================
  // Shower Shape Variables
  //=======================================================

  public :

    struct ShowerShape
     {
      float sigmaEtaEta ;        // weighted cluster rms along eta and inside 5x5 (absolute eta)
      float sigmaIetaIeta ;      // weighted cluster rms along eta and inside 5x5 (Xtal eta)
      float sigmaIphiIphi ;      // weighted cluster rms along phi and inside 5x5 (Xtal phi)
      float e1x5 ;               // energy inside 1x5 in etaxphi around the seed Xtal
      float e2x5Max ;            // energy inside 2x5 in etaxphi around the seed Xtal (max bwt the 2 possible sums)
      float e5x5 ;               // energy inside 5x5 in etaxphi around the seed Xtal
      float r9 ;                 // ratio of the 3x3 energy and supercluster energy
      float hcalDepth1OverEcal ; // hcal over ecal seed cluster energy using 1st hcal depth (using hcal towers within a cone)
      float hcalDepth2OverEcal ; // hcal over ecal seed cluster energy using 2nd hcal depth (using hcal towers within a cone)
      std::vector<CaloTowerDetId> hcalTowersBehindClusters ; //
      float hcalDepth1OverEcalBc ; // hcal over ecal seed cluster energy using 1st hcal depth (using hcal towers behind clusters)
      float hcalDepth2OverEcalBc ; // hcal over ecal seed cluster energy using 2nd hcal depth (using hcal towers behind clusters)
      ShowerShape()
       : sigmaEtaEta(std::numeric_limits<float>::max()),
       sigmaIetaIeta(std::numeric_limits<float>::max()),
       sigmaIphiIphi(std::numeric_limits<float>::max()),
	     e1x5(0.), e2x5Max(0.), e5x5(0.),
	     r9(-std::numeric_limits<float>::max()),
       hcalDepth1OverEcal(0.), hcalDepth2OverEcal(0.),
       hcalDepth1OverEcalBc(0.), hcalDepth2OverEcalBc(0.)
       {}
     } ;

    // accessors
    float sigmaEtaEta() const { return showerShape_.sigmaEtaEta ; }
    float sigmaIetaIeta() const { return showerShape_.sigmaIetaIeta ; }
    float sigmaIphiIphi() const { return showerShape_.sigmaIphiIphi ; }
    float e1x5() const { return showerShape_.e1x5 ; }
    float e2x5Max() const { return showerShape_.e2x5Max ; }
    float e5x5() const { return showerShape_.e5x5 ; }
    float r9() const { return showerShape_.r9 ; }
    float hcalDepth1OverEcal() const { return showerShape_.hcalDepth1OverEcal ; }
    float hcalDepth2OverEcal() const { return showerShape_.hcalDepth2OverEcal ; }
    float hcalOverEcal() const { return hcalDepth1OverEcal() + hcalDepth2OverEcal() ; }
    const std::vector<CaloTowerDetId> & hcalTowersBehindClusters() const { return showerShape_.hcalTowersBehindClusters ; }
    float hcalDepth1OverEcalBc() const { return showerShape_.hcalDepth1OverEcalBc ; }
    float hcalDepth2OverEcalBc() const { return showerShape_.hcalDepth2OverEcalBc ; }
    float hcalOverEcalBc() const { return hcalDepth1OverEcalBc() + hcalDepth2OverEcalBc() ; }
    const ShowerShape & showerShape() const { return showerShape_ ; }

    // setters (if you know what you're doing)
    void setShowerShape(const ShowerShape &s) { showerShape_ = s; }

    // for backward compatibility
    float scSigmaEtaEta() const { return sigmaEtaEta() ; }
    float scSigmaIEtaIEta() const { return sigmaIetaIeta() ; }
    float scE1x5() const { return e1x5() ; }
    float scE2x5Max() const { return e2x5Max() ; }
    float scE5x5() const { return e5x5() ; }
    float hadronicOverEm() const {return hcalOverEcal();}
    float hadronicOverEm1() const {return hcalDepth1OverEcal();}
    float hadronicOverEm2() const {return hcalDepth2OverEcal();}


  private:

    // attributes
    ShowerShape showerShape_ ;


  //=======================================================
  // Isolation Variables
  //=======================================================

  public :

    struct IsolationVariables
     {
      float tkSumPt ;                // track iso deposit with electron footprint removed
      float ecalRecHitSumEt ;        // ecal iso deposit with electron footprint removed
      float hcalDepth1TowerSumEt ;   // hcal depht 1 iso deposit with electron footprint removed
      float hcalDepth2TowerSumEt ;   // hcal depht 2 iso deposit with electron footprint removed
      float hcalDepth1TowerSumEtBc ; // hcal depht 1 iso deposit without towers behind clusters
      float hcalDepth2TowerSumEtBc ; // hcal depht 2 iso deposit without towers behind clusters
      IsolationVariables()
       : tkSumPt(0.), ecalRecHitSumEt(0.),
         hcalDepth1TowerSumEt(0.), hcalDepth2TowerSumEt(0.),
         hcalDepth1TowerSumEtBc(0.), hcalDepth2TowerSumEtBc(0.)
       {}
     } ;

    // 03 accessors
    float dr03TkSumPt() const { return dr03_.tkSumPt ; }
    float dr03EcalRecHitSumEt() const { return dr03_.ecalRecHitSumEt ; }
    float dr03HcalDepth1TowerSumEt() const { return dr03_.hcalDepth1TowerSumEt ; }
    float dr03HcalDepth2TowerSumEt() const { return dr03_.hcalDepth2TowerSumEt ; }
    float dr03HcalTowerSumEt() const { return dr03HcalDepth1TowerSumEt()+dr03HcalDepth2TowerSumEt() ; }
    float dr03HcalDepth1TowerSumEtBc() const { return dr03_.hcalDepth1TowerSumEtBc ; }
    float dr03HcalDepth2TowerSumEtBc() const { return dr03_.hcalDepth2TowerSumEtBc ; }
    float dr03HcalTowerSumEtBc() const { return dr03HcalDepth1TowerSumEtBc()+dr03HcalDepth2TowerSumEtBc() ; }
    const IsolationVariables & dr03IsolationVariables() const { return dr03_ ; }

    // 04 accessors
    float dr04TkSumPt() const { return dr04_.tkSumPt ; }
    float dr04EcalRecHitSumEt() const { return dr04_.ecalRecHitSumEt ; }
    float dr04HcalDepth1TowerSumEt() const { return dr04_.hcalDepth1TowerSumEt ; }
    float dr04HcalDepth2TowerSumEt() const { return dr04_.hcalDepth2TowerSumEt ; }
    float dr04HcalTowerSumEt() const { return dr04HcalDepth1TowerSumEt()+dr04HcalDepth2TowerSumEt() ; }
    float dr04HcalDepth1TowerSumEtBc() const { return dr04_.hcalDepth1TowerSumEtBc ; }
    float dr04HcalDepth2TowerSumEtBc() const { return dr04_.hcalDepth2TowerSumEtBc ; }
    float dr04HcalTowerSumEtBc() const { return dr04HcalDepth1TowerSumEtBc()+dr04HcalDepth2TowerSumEtBc() ; }
    const IsolationVariables & dr04IsolationVariables() const { return dr04_ ; }

    // setters ?!?
    void setDr03Isolation( const IsolationVariables & dr03 ) { dr03_ = dr03 ; }
    void setDr04Isolation( const IsolationVariables & dr04 ) { dr04_ = dr04 ; }

    // for backward compatibility
    void setIsolation03( const IsolationVariables & dr03 ) { dr03_ = dr03 ; }
    void setIsolation04( const IsolationVariables & dr04 ) { dr04_ = dr04 ; }
    const IsolationVariables & isolationVariables03() const { return dr03_ ; }
    const IsolationVariables & isolationVariables04() const { return dr04_ ; }

  private:

    // attributes
    IsolationVariables dr03_ ;
    IsolationVariables dr04_ ;


  //=======================================================
  // Conversion Rejection Information
  //=======================================================

  public :

    struct ConversionRejection
     {
      int flags ;  // -max:not-computed, other: as computed by Puneeth conversion code
      TrackBaseRef partner ; // conversion partner
      float dist ; // distance to the conversion partner
      float dcot ; // difference of cot(angle) with the conversion partner track
      float radius ; // signed conversion radius
      ConversionRejection()
       : flags(-1),
         dist(std::numeric_limits<float>::max()),
         dcot(std::numeric_limits<float>::max()),
         radius(std::numeric_limits<float>::max())
       {}
     } ;

    // accessors
    int convFlags() const { return conversionRejection_.flags ; }
    TrackBaseRef convPartner() const { return conversionRejection_.partner ; }
    float convDist() const { return conversionRejection_.dist ; }
    float convDcot() const { return conversionRejection_.dcot ; }
    float convRadius() const { return conversionRejection_.radius ; }
    const ConversionRejection & conversionRejectionVariables() const { return conversionRejection_ ; }

  private:

    // attributes
    ConversionRejection conversionRejection_ ;


  //=======================================================
  // Pflow Information
  //=======================================================

  public:

    struct PflowIsolationVariables
      {
       //first three data members that changed names, according to DataFormats/MuonReco/interface/MuonPFIsolation.h
       float sumChargedHadronPt; //!< sum-pt of charged Hadron    // old float chargedHadronIso ;
       float sumNeutralHadronEt;  //!< sum pt of neutral hadrons  // old float neutralHadronIso ;
       float sumPhotonEt;  //!< sum pt of PF photons              // old float photonIso ;
       //then four new data members, corresponding to DataFormats/MuonReco/interface/MuonPFIsolation.h
       float sumChargedParticlePt; //!< sum-pt of charged Particles(inludes e/mu) 
       float sumNeutralHadronEtHighThreshold;  //!< sum pt of neutral hadrons with a higher threshold
       float sumPhotonEtHighThreshold;  //!< sum pt of PF photons with a higher threshold
       float sumPUPt;  //!< sum pt of charged Particles not from PV  (for Pu corrections)

       PflowIsolationVariables() :
        sumChargedHadronPt(0),sumNeutralHadronEt(0),sumPhotonEt(0),sumChargedParticlePt(0),
        sumNeutralHadronEtHighThreshold(0),sumPhotonEtHighThreshold(0),sumPUPt(0) {}; 
      } ;

    struct MvaInput
     {
      int earlyBrem ; // Early Brem detected (-2=>unknown,-1=>could not be evaluated,0=>wrong,1=>true)
      int lateBrem ; // Late Brem detected (-2=>unknown,-1=>could not be evaluated,0=>wrong,1=>true)
      float sigmaEtaEta ; // Sigma-eta-eta with the PF cluster
      float hadEnergy ; // Associated PF Had Cluster energy
      float deltaEta ; // PF-cluster GSF track delta-eta
      int nClusterOutsideMustache ; // -2 => unknown, -1 =>could not be evaluated, 0 and more => number of clusters
      float etOutsideMustache ;
      MvaInput()
       : earlyBrem(-2), lateBrem(-2),
         sigmaEtaEta(std::numeric_limits<float>::max()),
         hadEnergy(0.),
         deltaEta(std::numeric_limits<float>::max()),
         nClusterOutsideMustache(-2),
         etOutsideMustache(-std::numeric_limits<float>::max())
       {}
     } ;

    struct MvaOutput
     {
      int status ; // see PFCandidateElectronExtra::StatusFlag
      float mva ;
      float mvaByPassForIsolated ; // complementary MVA used in preselection
      MvaOutput()
       : status(-1), mva(-999999999.), mvaByPassForIsolated(-999999999.)
       {}
     } ;

    // accessors
    const ShowerShape & pfShowerShape() const { return pfShowerShape_ ; }
    const PflowIsolationVariables & pfIsolationVariables() const { return pfIso_ ; }
    const MvaInput & mvaInput() const { return mvaInput_ ; }
    const MvaOutput & mvaOutput() const { return mvaOutput_ ; }

    // setters
    void setPfShowerShape( const ShowerShape & shape ) { pfShowerShape_ = shape ; }
    void setPfIsolationVariables( const PflowIsolationVariables & iso ) { pfIso_ = iso ; }
    void setMvaInput( const MvaInput & mi ) { mvaInput_ = mi ; }
    void setMvaOutput( const MvaOutput & mo ) { mvaOutput_ = mo ; }

    // for backward compatibility
    float mva() const { return mvaOutput_.mva ; }

  private:

    ShowerShape pfShowerShape_ ;
    PflowIsolationVariables pfIso_ ;
    MvaInput mvaInput_ ;
    MvaOutput mvaOutput_ ;


  //=======================================================
  // Preselection and Ambiguity
  //=======================================================

  public :

    // accessors
    bool ecalDriven() const ; // return true if ecalDrivenSeed() and passingCutBasedPreselection()
    bool passingCutBasedPreselection() const { return passCutBasedPreselection_ ; }
    bool passingPflowPreselection() const { return passPflowPreselection_ ; }
    bool ambiguous() const { return ambiguous_ ; }
    GsfTrackRefVector::size_type ambiguousGsfTracksSize() const { return ambiguousGsfTracks_.size() ; }
    GsfTrackRefVector::const_iterator ambiguousGsfTracksBegin() const { return ambiguousGsfTracks_.begin() ; }
    GsfTrackRefVector::const_iterator ambiguousGsfTracksEnd() const { return ambiguousGsfTracks_.end() ; }

    // setters
    void setPassCutBasedPreselection( bool flag ) { passCutBasedPreselection_ = flag ; }
    void setPassPflowPreselection( bool flag ) { passPflowPreselection_ = flag ; }
    void setAmbiguous( bool flag ) { ambiguous_ = flag ; }
    void clearAmbiguousGsfTracks() { ambiguousGsfTracks_.clear() ; }
    void addAmbiguousGsfTrack( const reco::GsfTrackRef & t ) { ambiguousGsfTracks_.push_back(t) ; }

    // backward compatibility
    void setPassMvaPreselection( bool flag ) { passMvaPreslection_ = flag ; }
    bool passingMvaPreselection() const { return passMvaPreslection_ ; }

  private:

    // attributes
    bool passCutBasedPreselection_ ;
    bool passPflowPreselection_ ;
    bool passMvaPreslection_ ; // to be removed : passPflowPreslection_
    bool ambiguous_ ;
    GsfTrackRefVector ambiguousGsfTracks_ ; // ambiguous gsf tracks


  //=======================================================
  // Brem Fractions and Classification
  //=======================================================

  public :

    struct ClassificationVariables
      {
       float trackFbrem  ;       // the brem fraction from gsf fit: (track momentum in - track momentum out) / track momentum in
       float superClusterFbrem ; // the brem fraction from supercluster: (supercluster energy - electron cluster energy) / supercluster energy
       float pfSuperClusterFbrem ; // the brem fraction from pflow supercluster
       ClassificationVariables()
        : trackFbrem(-1.e30), superClusterFbrem(-1.e30), pfSuperClusterFbrem(-1.e30)
        {}
      } ;
    enum Classification { UNKNOWN=-1, GOLDEN=0, BIGBREM=1, BADTRACK=2, SHOWERING=3, GAP=4 } ;

    // accessors
    float trackFbrem() const { return classVariables_.trackFbrem ; }
    float superClusterFbrem() const { return classVariables_.superClusterFbrem ; }
    float pfSuperClusterFbrem() const { return classVariables_.pfSuperClusterFbrem ; }
    const ClassificationVariables & classificationVariables() const { return classVariables_ ; }
    Classification classification() const { return class_ ; }

    // utilities
    int numberOfBrems() const { return basicClustersSize()-1 ; }
    float fbrem() const { return trackFbrem() ; }

    // setters
    void setTrackFbrem( float fbrem ) { classVariables_.trackFbrem = fbrem ; }
    void setSuperClusterFbrem( float fbrem ) { classVariables_.superClusterFbrem = fbrem ; }
    void setPfSuperClusterFbrem( float fbrem ) { classVariables_.pfSuperClusterFbrem = fbrem ; }
    void setClassificationVariables( const ClassificationVariables & cv ) { classVariables_ = cv ; }
    void setClassification( Classification myclass ) { class_ = myclass ; }

  private:

    // attributes
    ClassificationVariables classVariables_ ;
    Classification class_ ; // fbrem and number of clusters based electron classification


  //=======================================================
  // Corrections
  //
  // The only methods, with classification, which modify
  // the electrons after they have been constructed.
  // They change a given characteristic, such as the super-cluster
  // energy, and propagate the change consistently
  // to all the depending attributes.
  // We expect the methods to be called in a given order
  // and so to store specific kind of corrections
  // 1) classify()
  // 2) correctEcalEnergy() : depending on classification and eta
  // 3) correctMomemtum() : depending on classification and ecal energy and tracker momentum errors
  //
  // Beware that correctEcalEnergy() is modifying few attributes which
  // were potentially used for preselection, whose value used in
  // preselection will not be available any more :
  // hcalDepth1OverEcal, hcalDepth2OverEcal, eSuperClusterOverP,
  // eSeedClusterOverP, eEleClusterOverPout.
  //=======================================================

  public :

    enum P4Kind { P4_UNKNOWN=-1, P4_FROM_SUPER_CLUSTER=0, P4_COMBINATION=1, P4_PFLOW_COMBINATION=2 } ;

    struct Corrections
     {
      bool  isEcalEnergyCorrected ;    // true if ecal energy has been corrected
      float correctedEcalEnergy ;      // corrected energy (if !isEcalEnergyCorrected this value is identical to the supercluster energy)
      float correctedEcalEnergyError ; // error on energy
      //bool isMomentumCorrected ;     // DEPRECATED
      float trackMomentumError ;       // track momentum error from gsf fit
      //
      LorentzVector fromSuperClusterP4 ; // for P4_FROM_SUPER_CLUSTER
      float fromSuperClusterP4Error ;    // for P4_FROM_SUPER_CLUSTER
      LorentzVector combinedP4 ;    // for P4_COMBINATION
      float combinedP4Error ;       // for P4_COMBINATION
      LorentzVector pflowP4 ;       // for P4_PFLOW_COMBINATION
      float pflowP4Error ;          // for P4_PFLOW_COMBINATION
      P4Kind candidateP4Kind ;  // say which momentum has been stored in reco::Candidate
      //
      Corrections()
       : isEcalEnergyCorrected(false), correctedEcalEnergy(0.), correctedEcalEnergyError(999.),
  	     /*isMomentumCorrected(false),*/ trackMomentumError(999.),
  	     fromSuperClusterP4Error(999.), combinedP4Error(999.), pflowP4Error(999.),
  	     candidateP4Kind(P4_UNKNOWN)
       {}
     } ;

    // setters
    void setCorrectedEcalEnergyError( float newEnergyError ) ;
    void setCorrectedEcalEnergy( float newEnergy ) ;
    void setTrackMomentumError( float trackMomentumError ) ;
    void setP4( P4Kind kind, const LorentzVector & p4, float p4Error, bool setCandidate ) ;
    using RecoCandidate::setP4 ;

    // accessors
    bool isEcalEnergyCorrected() const { return corrections_.isEcalEnergyCorrected ; }
    float correctedEcalEnergy() const { return corrections_.correctedEcalEnergy ; }
    float correctedEcalEnergyError() const { return corrections_.correctedEcalEnergyError ; }
    float trackMomentumError() const { return corrections_.trackMomentumError ; }
    const LorentzVector & p4( P4Kind kind ) const ;
    using RecoCandidate::p4 ;
    float p4Error( P4Kind kind ) const ;
    P4Kind candidateP4Kind() const { return corrections_.candidateP4Kind ; }
    const Corrections & corrections() const { return corrections_ ; }
    
    // bare setter (if you know what you're doing)
    void setCorrections(const Corrections &c) { corrections_ = c; }

    // for backward compatibility
    void setEcalEnergyError( float energyError ) { setCorrectedEcalEnergyError(energyError) ; }
    float ecalEnergy() const { return correctedEcalEnergy() ; }
    float ecalEnergyError() const { return correctedEcalEnergyError() ; }
    //bool isMomentumCorrected() const { return corrections_.isMomentumCorrected ; }
    float caloEnergy() const { return correctedEcalEnergy() ; }
    bool isEnergyScaleCorrected() const { return isEcalEnergyCorrected() ; }
    void correctEcalEnergy( float newEnergy, float newEnergyError )
     {
      setCorrectedEcalEnergy(newEnergy) ;
      setEcalEnergyError(newEnergyError) ;
     }
    void correctMomentum( const LorentzVector & p4, float trackMomentumError, float p4Error )
     { setTrackMomentumError(trackMomentumError) ; setP4(P4_COMBINATION,p4,p4Error,true) ; }


  private:

    // attributes
    Corrections corrections_ ;

 } ;

 } // namespace reco

#endif
