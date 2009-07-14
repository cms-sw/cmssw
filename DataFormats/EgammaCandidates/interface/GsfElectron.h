#ifndef GsfElectron_h
#define GsfElectron_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
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
 * \author Ursula Berthon - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 * \version $Id: GsfElectron.h,v 1.29 2009/04/06 11:18:05 chamont Exp $
 *
 ****************************************************************************/

//*****************************************************************************
//
// $Log: GsfElectron.h,v $
// Revision 1.29  2009/04/06 11:18:05  chamont
// few changes, should not affect users
//
// Revision 1.28  2009/03/31 10:54:09  charlot
// readded isolation setters
//
// Revision 1.27  2009/03/31 10:30:20  charlot
// added getters for structs, removed setter for the core
//
// Revision 1.26  2009/03/28 20:42:11  charlot
// added momentum at vertex with bs constraint
//
// Revision 1.25  2009/03/27 13:09:08  charlot
// added setter for the core
//
// Revision 1.24  2009/03/26 11:20:20  charlot
// updated for new supercluster dataformat
//
// Revision 1.23  2009/03/24 23:07:03  charlot
// added setter for isolation
//
// Revision 1.22  2009/03/24 17:26:27  charlot
// updated provenance and added comments in headers
//
// Revision 1.21  2009/03/20 22:59:16  chamont
// new class GsfElectronCore and new interface for GsfElectron
//
// Revision 1.20  2009/02/14 11:00:26  charlot
// new interface for fiducial regions
//
//*****************************************************************************

class GsfElectron : public RecoCandidate
 {

  //=======================================================
  // Constructors
  //=======================================================

  public :

    // some nested structures defined later on
    struct TrackClusterMatching ;
    struct TrackExtrapolations ;
    struct ClosestCtfTrack ;
    struct FiducialFlags ;
    struct ShowerShape ;
    struct IsolationVariables ;

    GsfElectron() ;
    GsfElectron
     (
      const LorentzVector & p4, int charge, const GsfElectronCoreRef &,
      const TrackClusterMatching &, const TrackExtrapolations &, const ClosestCtfTrack &,
      const FiducialFlags &, const ShowerShape &, float fbrem, float mva
     ) ;
    GsfElectron * clone() const ;
    virtual ~GsfElectron() {} ;


  //=======================================================
  // Candidate Methods
  //
  // GsfElectron inherits from RecoCandidate, thus it
  // implements the Candidate methods, such as p4().
  //=======================================================

  public:

    bool isElectron() const { return true ; }

  protected :

    virtual bool overlap( const Candidate & ) const ;

  //=======================================================
  // Core Attributes
  //
  // They all have been computed before, when building the
  // collection of GsfElectronCore instances. Each GsfElectron
  // has a reference toward a GsfElectronCore.
  //=======================================================

  public:

    // accessors
    GsfElectronCoreRef core() const { return core_ ; }

    // forward core methods
    SuperClusterRef superCluster() const { return core_->superCluster() ; }
    GsfTrackRef gsfTrack() const { return core_->gsfTrack() ; }
    bool isEcalDriven() const { return core_->isEcalDriven() ; }
    bool isTrackerDriven() const { return core_->isTrackerDriven() ; }
    SuperClusterRef pflowSuperCluster() const { return core_->pflowSuperCluster() ; }
    int trackCharge() const { return core_->gsfTrack()->charge() ; }

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
      float deltaPhiEleClusterAtCalo ;  // the supercluster phi - track phi position at calo extrapolated from the innermost track state
      float deltaPhiSuperClusterAtVtx ; // the seed cluster phi - track phi position at calo extrapolated from the outermost track state
      float deltaPhiSeedClusterAtCalo ; // the electron cluster phi - track phi position at calo extrapolated from the outermost track state
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
      math::XYZPoint  positionAtVtx ;     // the track PCA to the beam spot
      math::XYZPoint  positionAtCalo ;    // the track PCA to the supercluster position
      math::XYZVector momentumAtVtx ;     // the track momentum at the PCA to the beam spot
      math::XYZVector momentumAtCalo ;    // the track momentum extrapolated at the supercluster position from the innermost track state
      math::XYZVector momentumOut ;       // the track momentum extrapolated at the seed cluster position from the outermost track state
      math::XYZVector momentumAtEleClus ; // the track momentum extrapolated at the ele cluster position from the outermost track state
      math::XYZVector momentumAtVtxWithConstraint ;     // the track momentum at the PCA to the beam spot using bs constraint
     } ;

    // accessors
    math::XYZPoint trackPositionAtVtx() const { return trackExtrapolations_.positionAtVtx ; }
    math::XYZPoint trackPositionAtCalo() const { return trackExtrapolations_.positionAtCalo ; }
    math::XYZVector trackMomentumAtVtx() const { return trackExtrapolations_.momentumAtVtx ; }
    math::XYZVector trackMomentumAtCalo() const { return trackExtrapolations_.momentumAtCalo ; }
    math::XYZVector trackMomentumOut() const { return trackExtrapolations_.momentumOut ; }
    math::XYZVector trackMomentumAtEleClus() const { return trackExtrapolations_.momentumAtEleClus ; }
    math::XYZVector trackMomentumAtVtxWithConstraint() const { return trackExtrapolations_.momentumAtVtxWithConstraint ; }
    const TrackExtrapolations & trackExtrapolations() const { return trackExtrapolations_ ; }

    // for backward compatibility
    math::XYZPoint TrackPositionAtVtx() const { return trackPositionAtVtx() ; }
    math::XYZPoint TrackPositionAtCalo() const { return trackPositionAtCalo() ; }


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
  // Other tracks
  //=======================================================

  public :

    struct ClosestCtfTrack
     {
      TrackRef ctfTrack ; // best matching ctf track
      float shFracInnerHits ; // fraction of common hits between the ctf and gsf tracks
  	  ClosestCtfTrack() : shFracInnerHits(0.) {}
     } ;

    // accessors
    TrackRef closestCtfTrackRef() const { return closestCtfTrack_.ctfTrack ; } // get the CTF track best matching the GTF associated to this electron
    float shFracInnerHits() const { return closestCtfTrack_.shFracInnerHits ; } // measure the fraction of common hits between the GSF and CTF tracks
    const ClosestCtfTrack & closestCtfTrack() const { return closestCtfTrack_ ; }
    GsfTrackRefVector::size_type ambiguousGsfTracksSize() const { return ambiguousGsfTracks_.size() ; }
    GsfTrackRefVector::const_iterator ambiguousGsfTracksBegin() const { return ambiguousGsfTracks_.begin() ; }
    GsfTrackRefVector::const_iterator ambiguousGsfTracksEnd() const { return ambiguousGsfTracks_.end() ; }

    // setters
    void addAmbiguousGsfTrack( const reco::GsfTrackRef & t ) { ambiguousGsfTracks_.push_back(t) ; }


  private:

    // attributes
    ClosestCtfTrack closestCtfTrack_ ;
    GsfTrackRefVector ambiguousGsfTracks_ ; // ambiguous gsf tracks


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
      float sigmaIetaIeta ;      // weighted cluster rms along eta and inside 5x5 (new, Xtal eta)
      float e1x5 ;               // energy inside 1x5 in etaxphi around the seed Xtal
      float e2x5Max ;            // energy inside 2x5 in etaxphi around the seed Xtal (max bwt the 2 possible sums)
      float e5x5 ;               // energy inside 5x5 in etaxphi around the seed Xtal
      float hcalDepth1OverEcal ; // hcal over ecal seed cluster energy using first hcal depth (hcal is energy of towers within dR=015)
      float hcalDepth2OverEcal ; // hcal over ecal seed cluster energy using 2nd hcal depth (hcal is energy of towers within dR=015)
      ShowerShape()
       : sigmaEtaEta(std::numeric_limits<float>::infinity()),
	     sigmaIetaIeta(std::numeric_limits<float>::infinity()),
	     e1x5(0.), e2x5Max(0.), e5x5(0.),
	     hcalDepth1OverEcal(0), hcalDepth2OverEcal(0)
       {}
     } ;

    // accessors
    float sigmaEtaEta() const { return showerShape_.sigmaEtaEta ; }
    float sigmaIetaIeta() const { return showerShape_.sigmaIetaIeta ; }
    float e1x5() const { return showerShape_.e1x5 ; }
    float e2x5Max() const { return showerShape_.e2x5Max ; }
    float e5x5() const { return showerShape_.e5x5 ; }
    float hcalDepth1OverEcal() const { return showerShape_.hcalDepth1OverEcal ; }
    float hcalDepth2OverEcal() const { return showerShape_.hcalDepth2OverEcal ; }
    float hcalOverEcal() const { return hcalDepth1OverEcal() + hcalDepth2OverEcal() ; }
    const ShowerShape & showerShape() const { return showerShape_ ; }

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
      IsolationVariables()
       : tkSumPt(0.), ecalRecHitSumEt(0.), hcalDepth1TowerSumEt(0.), hcalDepth2TowerSumEt(0.)
       {}
     } ;

    // 03 accessors
    float dr03TkSumPt() const { return dr03_.tkSumPt ; }
    float dr03EcalRecHitSumEt() const { return dr03_.ecalRecHitSumEt ; }
    float dr03HcalDepth1TowerSumEt() const { return dr03_.hcalDepth1TowerSumEt ; }
    float dr03HcalDepth2TowerSumEt() const { return dr03_.hcalDepth2TowerSumEt ; }
    float dr03HcalTowerSumEt() const { return dr03HcalDepth1TowerSumEt()+dr03HcalDepth2TowerSumEt() ; }
    const IsolationVariables & dr03IsolationVariables() const { return dr03_ ; }

    // 04 accessors
    float dr04TkSumPt() const { return dr04_.tkSumPt ; }
    float dr04EcalRecHitSumEt() const { return dr04_.ecalRecHitSumEt ; }
    float dr04HcalDepth1TowerSumEt() const { return dr04_.hcalDepth1TowerSumEt ; }
    float dr04HcalDepth2TowerSumEt() const { return dr04_.hcalDepth2TowerSumEt ; }
    float dr04HcalTowerSumEt() const { return dr04HcalDepth1TowerSumEt()+dr04HcalDepth2TowerSumEt() ; }
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
  // Particle Flow Data
  //=======================================================

  public :

    // accessors
    float mva() const { return mva_ ; }

  private:

    // attributes
    float mva_ ;                    // electron ID variable from mva (tracker driven electrons)


  //=======================================================
  // Brem Fraction and Classification
  // * fbrem given to the GsfElectron constructor
  // * classification computed later
  //=======================================================

  public :

    enum Classification { UNKNOWN =-1, GOLDEN, BIGBREM, NARROW, SHOWERING, GAP } ;

    // accessors
    float fbrem() const { return fbrem_ ; }
    int numberOfBrems() const { return basicClustersSize()-1 ; }
    Classification classification() const { return class_ ; }

    // setters
    void classifyElectron( Classification myclass ) { class_ = myclass ; }

  private:

    // attributes
    float fbrem_ ; // the brem fraction from gsf fit: (track momentum in - track momentum out) / track momentum in
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
  // 3) correctMomemtum() : depending on classification and acal energy and tracker momentum errors
  //
  // Beware that correctEcalEnergy() is modifying few attributes which
  // were potentially used for preselection, whose value used in
  // preselection will not be available any more :
  // hcalDepth1OverEcal, hcalDepth2OverEcal, eSuperClusterOverP,
  // eSeedClusterOverP, eEleClusterOverPout.
  //=======================================================

  public :

    struct Corrections
     {
      bool isEcalEnergyCorrected ;  // true if ecal energy has been corrected
      float ecalEnergy ;            // ecal corrected energy (if !isEcalEnergyCorrected this value is identical to the supercluster energy)
      float ecalEnergyError ;       // error on correctedCaloEnergy
      bool isMomentumCorrected ;    // true if E-p combination has been applied (if not the electron momentum is the ecal corrected energy)
      float trackMomentumError ;    // track momentum error from gsf fit
      float electronMomentumError ; // the final electron momentum error
      Corrections()
       : isEcalEnergyCorrected(false), ecalEnergy(0.), ecalEnergyError(999.),
  	     isMomentumCorrected(false), trackMomentumError(999.), electronMomentumError(999.)
       {}
     } ;

    // correctors
    void correctEcalEnergy( float newEnergy, float newEnergyError ) ;
    void correctMomentum
     ( const LorentzVector & momentum,
       float trackMomentumError, float electronMomentumError ) ;

    // accessors
    bool isEcalEnergyCorrected() const { return corrections_.isEcalEnergyCorrected ; }
    float ecalEnergy() const { return corrections_.ecalEnergy ; }
    float ecalEnergyError() const { return corrections_.ecalEnergyError ; }
    bool isMomentumCorrected() const { return corrections_.isMomentumCorrected ; }
    float trackMomentumError() const { return corrections_.trackMomentumError ; }
    float electronMomentumError() const { return corrections_.electronMomentumError ; }
    const Corrections & corrections() const { return corrections_ ; }

    // for backward compatibility
    float caloEnergy() const { return ecalEnergy() ; }
    //void correctElectronEnergyScale( const float newEnergy )
    // { correctEcalEnergy(newEnergy) ; }
    //void correctElectronFourMomentum
    // ( const LorentzVector & m,
    //   float & enErr, float  & tMerr)
    // { correctMomentum(m,enErr,tMerr,0) ; }
    bool isEnergyScaleCorrected() const { return isEcalEnergyCorrected() ; }

  private:

    // attributes
    Corrections corrections_ ;

 } ;

  //typedef GsfElectron PixelMatchGsfElectron ;

 }

#endif
