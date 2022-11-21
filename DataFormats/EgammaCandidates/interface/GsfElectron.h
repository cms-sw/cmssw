#ifndef GsfElectron_h
#define GsfElectron_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
//#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <vector>
#include <limits>
#include <numeric>

namespace reco {

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

  class GsfElectron : public RecoCandidate {
    //=======================================================
    // Constructors
    //
    // The clone() method with arguments, and the copy
    // constructor with edm references is designed for
    // someone which want to duplicates all
    // collections.
    //=======================================================

  public:
    // some nested structures defined later on
    struct ChargeInfo;
    struct TrackClusterMatching;
    struct TrackExtrapolations;
    struct ClosestCtfTrack;
    struct FiducialFlags;
    struct ShowerShape;
    struct IsolationVariables;
    struct ConversionRejection;
    struct ClassificationVariables;
    struct SaturationInfo;

    GsfElectron();
    GsfElectron(const GsfElectronCoreRef &);
    GsfElectron(const GsfElectron &, const GsfElectronCoreRef &);
    GsfElectron(const GsfElectron &electron,
                const GsfElectronCoreRef &core,
                const CaloClusterPtr &electronCluster,
                const TrackRef &closestCtfTrack,
                const TrackBaseRef &conversionPartner,
                const GsfTrackRefVector &ambiguousTracks);
    GsfElectron(int charge,
                const ChargeInfo &,
                const GsfElectronCoreRef &,
                const TrackClusterMatching &,
                const TrackExtrapolations &,
                const ClosestCtfTrack &,
                const FiducialFlags &,
                const ShowerShape &,
                const ConversionRejection &);
    GsfElectron(int charge,
                const ChargeInfo &,
                const GsfElectronCoreRef &,
                const TrackClusterMatching &,
                const TrackExtrapolations &,
                const ClosestCtfTrack &,
                const FiducialFlags &,
                const ShowerShape &,
                const ShowerShape &,
                const ConversionRejection &,
                const SaturationInfo &);
    GsfElectron *clone() const override;
    GsfElectron *clone(const GsfElectronCoreRef &core,
                       const CaloClusterPtr &electronCluster,
                       const TrackRef &closestCtfTrack,
                       const TrackBaseRef &conversionPartner,
                       const GsfTrackRefVector &ambiguousTracks) const;
    ~GsfElectron() override{};

  private:
    void init();

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
    struct ChargeInfo {
      int scPixCharge;
      bool isGsfCtfScPixConsistent;
      bool isGsfScPixConsistent;
      bool isGsfCtfConsistent;
      ChargeInfo()
          : scPixCharge(0), isGsfCtfScPixConsistent(false), isGsfScPixConsistent(false), isGsfCtfConsistent(false) {}
    };

    // Charge info accessors
    // to get gsf track charge: gsfTrack()->charge()
    // to get ctf track charge, if closestCtfTrackRef().isNonnull(): closestCtfTrackRef()->charge()
    int scPixCharge() const { return chargeInfo_.scPixCharge; }
    bool isGsfCtfScPixChargeConsistent() const { return chargeInfo_.isGsfCtfScPixConsistent; }
    bool isGsfScPixChargeConsistent() const { return chargeInfo_.isGsfScPixConsistent; }
    bool isGsfCtfChargeConsistent() const { return chargeInfo_.isGsfCtfConsistent; }
    const ChargeInfo &chargeInfo() const { return chargeInfo_; }

    // Candidate redefined methods
    bool isElectron() const override { return true; }
    bool overlap(const Candidate &) const override;

  private:
    // Complementary attributes
    ChargeInfo chargeInfo_;

    //=======================================================
    // Core Attributes
    //
    // They all have been computed before, when building the
    // collection of GsfElectronCore instances. Each GsfElectron
    // has a reference toward a GsfElectronCore.
    //=======================================================

  public:
    // accessors
    virtual GsfElectronCoreRef core() const;
    void setCore(const reco::GsfElectronCoreRef &core) { core_ = core; }

    // forward core methods
    SuperClusterRef superCluster() const override { return core()->superCluster(); }
    GsfTrackRef gsfTrack() const override { return core()->gsfTrack(); }
    float ctfGsfOverlap() const { return core()->ctfGsfOverlap(); }
    bool ecalDrivenSeed() const { return core()->ecalDrivenSeed(); }
    bool trackerDrivenSeed() const { return core()->trackerDrivenSeed(); }
    virtual SuperClusterRef parentSuperCluster() const { return core()->parentSuperCluster(); }
    bool closestCtfTrackRefValid() const {
      return closestCtfTrackRef().isAvailable() && closestCtfTrackRef().isNonnull();
    }
    //methods used for MVA variables
    float closestCtfTrackNormChi2() const {
      return closestCtfTrackRefValid() ? closestCtfTrackRef()->normalizedChi2() : 0;
    }
    int closestCtfTrackNLayers() const {
      return closestCtfTrackRefValid() ? closestCtfTrackRef()->hitPattern().trackerLayersWithMeasurement() : -1;
    }

    // backward compatibility
    struct ClosestCtfTrack {
      TrackRef ctfTrack;      // best matching ctf track
      float shFracInnerHits;  // fraction of common hits between the ctf and gsf tracks
      ClosestCtfTrack() : shFracInnerHits(0.) {}
      ClosestCtfTrack(TrackRef track, float sh) : ctfTrack(track), shFracInnerHits(sh) {}
    };
    float shFracInnerHits() const { return core()->ctfGsfOverlap(); }
    virtual TrackRef closestCtfTrackRef() const { return core()->ctfTrack(); }
    virtual ClosestCtfTrack closestCtfTrack() const {
      return ClosestCtfTrack(core()->ctfTrack(), core()->ctfGsfOverlap());
    }

  private:
    // attributes
    GsfElectronCoreRef core_;

    //=======================================================
    // Track-Cluster Matching Attributes
    //=======================================================

  public:
    struct TrackClusterMatching {
      CaloClusterPtr electronCluster;  // basic cluster best matching gsf track
      float eSuperClusterOverP;        // the supercluster energy / track momentum at the PCA to the beam spot
      float eSeedClusterOverP;         // the seed cluster energy / track momentum at the PCA to the beam spot
      float eSeedClusterOverPout;  // the seed cluster energy / track momentum at calo extrapolated from the outermost track state
      float eEleClusterOverPout;  // the electron cluster energy / track momentum at calo extrapolated from the outermost track state
      float deltaEtaSuperClusterAtVtx;  // the supercluster eta - track eta position at calo extrapolated from innermost track state
      float deltaEtaSeedClusterAtCalo;  // the seed cluster eta - track eta position at calo extrapolated from the outermost track state
      float deltaEtaEleClusterAtCalo;  // the electron cluster eta - track eta position at calo extrapolated from the outermost state
      float deltaPhiEleClusterAtCalo;  // the electron cluster phi - track phi position at calo extrapolated from the outermost track state
      float deltaPhiSuperClusterAtVtx;  // the supercluster phi - track phi position at calo extrapolated from the innermost track state
      float deltaPhiSeedClusterAtCalo;  // the seed cluster phi - track phi position at calo extrapolated from the outermost track state
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
            deltaPhiSeedClusterAtCalo(std::numeric_limits<float>::max()) {}
    };

    // accessors
    CaloClusterPtr electronCluster() const { return trackClusterMatching_.electronCluster; }
    float eSuperClusterOverP() const { return trackClusterMatching_.eSuperClusterOverP; }
    float eSeedClusterOverP() const { return trackClusterMatching_.eSeedClusterOverP; }
    float eSeedClusterOverPout() const { return trackClusterMatching_.eSeedClusterOverPout; }
    float eEleClusterOverPout() const { return trackClusterMatching_.eEleClusterOverPout; }
    float deltaEtaSuperClusterTrackAtVtx() const { return trackClusterMatching_.deltaEtaSuperClusterAtVtx; }
    float deltaEtaSeedClusterTrackAtCalo() const { return trackClusterMatching_.deltaEtaSeedClusterAtCalo; }
    float deltaEtaEleClusterTrackAtCalo() const { return trackClusterMatching_.deltaEtaEleClusterAtCalo; }
    float deltaPhiSuperClusterTrackAtVtx() const { return trackClusterMatching_.deltaPhiSuperClusterAtVtx; }
    float deltaPhiSeedClusterTrackAtCalo() const { return trackClusterMatching_.deltaPhiSeedClusterAtCalo; }
    float deltaPhiEleClusterTrackAtCalo() const { return trackClusterMatching_.deltaPhiEleClusterAtCalo; }
    float deltaEtaSeedClusterTrackAtVtx() const {
      return superCluster().isNonnull() && superCluster()->seed().isNonnull()
                 ? trackClusterMatching_.deltaEtaSuperClusterAtVtx - superCluster()->eta() +
                       superCluster()->seed()->eta()
                 : std::numeric_limits<float>::max();
    }
    const TrackClusterMatching &trackClusterMatching() const { return trackClusterMatching_; }

    // for backward compatibility, usefull ?
    void setDeltaEtaSuperClusterAtVtx(float de) { trackClusterMatching_.deltaEtaSuperClusterAtVtx = de; }
    void setDeltaPhiSuperClusterAtVtx(float dphi) { trackClusterMatching_.deltaPhiSuperClusterAtVtx = dphi; }

  private:
    // attributes
    TrackClusterMatching trackClusterMatching_;

    //=======================================================
    // Track extrapolations
    //=======================================================

  public:
    struct TrackExtrapolations {
      math::XYZPointF positionAtVtx;   // the track PCA to the beam spot
      math::XYZPointF positionAtCalo;  // the track PCA to the supercluster position
      math::XYZVectorF momentumAtVtx;  // the track momentum at the PCA to the beam spot
      // the track momentum extrapolated at the supercluster position from the innermost track state
      math::XYZVectorF momentumAtCalo;
      // the track momentum extrapolated at the seed cluster position from the outermost track state
      math::XYZVectorF momentumOut;
      // the track momentum extrapolated at the ele cluster position from the outermost track state
      math::XYZVectorF momentumAtEleClus;
      math::XYZVectorF momentumAtVtxWithConstraint;  // the track momentum at the PCA to the beam spot using bs constraint
    };

    // accessors
    math::XYZPointF trackPositionAtVtx() const { return trackExtrapolations_.positionAtVtx; }
    math::XYZPointF trackPositionAtCalo() const { return trackExtrapolations_.positionAtCalo; }
    math::XYZVectorF trackMomentumAtVtx() const { return trackExtrapolations_.momentumAtVtx; }
    math::XYZVectorF trackMomentumAtCalo() const { return trackExtrapolations_.momentumAtCalo; }
    math::XYZVectorF trackMomentumOut() const { return trackExtrapolations_.momentumOut; }
    math::XYZVectorF trackMomentumAtEleClus() const { return trackExtrapolations_.momentumAtEleClus; }
    math::XYZVectorF trackMomentumAtVtxWithConstraint() const {
      return trackExtrapolations_.momentumAtVtxWithConstraint;
    }
    const TrackExtrapolations &trackExtrapolations() const { return trackExtrapolations_; }

    // setter (if you know what you're doing)
    void setTrackExtrapolations(const TrackExtrapolations &te) { trackExtrapolations_ = te; }

    // for backward compatibility
    math::XYZPointF TrackPositionAtVtx() const { return trackPositionAtVtx(); }
    math::XYZPointF TrackPositionAtCalo() const { return trackPositionAtCalo(); }

  private:
    // attributes
    TrackExtrapolations trackExtrapolations_;

    //=======================================================
    // SuperCluster direct access
    //=======================================================

  public:
    // direct accessors
    math::XYZPoint superClusterPosition() const { return superCluster()->position(); }  // the super cluster position
    int basicClustersSize() const {
      return superCluster()->clustersSize();
    }  // number of basic clusters inside the supercluster
    CaloCluster_iterator basicClustersBegin() const { return superCluster()->clustersBegin(); }
    CaloCluster_iterator basicClustersEnd() const { return superCluster()->clustersEnd(); }

    // for backward compatibility
    math::XYZPoint caloPosition() const { return superCluster()->position(); }

    //=======================================================
    // Fiducial Flags
    //=======================================================

  public:
    struct FiducialFlags {
      bool isEB;         // true if particle is in ECAL Barrel
      bool isEE;         // true if particle is in ECAL Endcaps
      bool isEBEEGap;    // true if particle is in the crack between EB and EE
      bool isEBEtaGap;   // true if particle is in EB, and inside the eta gaps between modules
      bool isEBPhiGap;   // true if particle is in EB, and inside the phi gaps between modules
      bool isEEDeeGap;   // true if particle is in EE, and inside the gaps between dees
      bool isEERingGap;  // true if particle is in EE, and inside the gaps between rings
      FiducialFlags()
          : isEB(false),
            isEE(false),
            isEBEEGap(false),
            isEBEtaGap(false),
            isEBPhiGap(false),
            isEEDeeGap(false),
            isEERingGap(false) {}
    };

    // accessors
    bool isEB() const { return fiducialFlags_.isEB; }
    bool isEE() const { return fiducialFlags_.isEE; }
    bool isGap() const { return ((isEBEEGap()) || (isEBGap()) || (isEEGap())); }
    bool isEBEEGap() const { return fiducialFlags_.isEBEEGap; }
    bool isEBGap() const { return (isEBEtaGap() || isEBPhiGap()); }
    bool isEBEtaGap() const { return fiducialFlags_.isEBEtaGap; }
    bool isEBPhiGap() const { return fiducialFlags_.isEBPhiGap; }
    bool isEEGap() const { return (isEEDeeGap() || isEERingGap()); }
    bool isEEDeeGap() const { return fiducialFlags_.isEEDeeGap; }
    bool isEERingGap() const { return fiducialFlags_.isEERingGap; }
    const FiducialFlags &fiducialFlags() const { return fiducialFlags_; }
    // setters... not necessary in regular situations
    // but handy for late stage modifications of electron objects
    void setFFlagIsEB(const bool b) { fiducialFlags_.isEB = b; }
    void setFFlagIsEE(const bool b) { fiducialFlags_.isEE = b; }
    void setFFlagIsEBEEGap(const bool b) { fiducialFlags_.isEBEEGap = b; }
    void setFFlagIsEBEtaGap(const bool b) { fiducialFlags_.isEBEtaGap = b; }
    void setFFlagIsEBPhiGap(const bool b) { fiducialFlags_.isEBPhiGap = b; }
    void setFFlagIsEEDeeGap(const bool b) { fiducialFlags_.isEEDeeGap = b; }
    void setFFlagIsEERingGap(const bool b) { fiducialFlags_.isEERingGap = b; }

  private:
    // attributes
    FiducialFlags fiducialFlags_;

    //=======================================================
    // Shower Shape Variables
    //=======================================================

  public:
    struct ShowerShape {
      float sigmaEtaEta;    // weighted cluster rms along eta and inside 5x5 (absolute eta)
      float sigmaIetaIeta;  // weighted cluster rms along eta and inside 5x5 (Xtal eta)
      float sigmaIphiIphi;  // weighted cluster rms along phi and inside 5x5 (Xtal phi)
      float e1x5;           // energy inside 1x5 in etaxphi around the seed Xtal
      float e2x5Max;        // energy inside 2x5 in etaxphi around the seed Xtal (max bwt the 2 possible sums)
      float e5x5;           // energy inside 5x5 in etaxphi around the seed Xtal
      float r9;             // ratio of the 3x3 energy and supercluster energy
      float hcalDepth1OverEcal;  // run2 hcal over ecal seed cluster energy using 1st hcal depth (using hcal towers within a cone)
      float hcalDepth2OverEcal;  // run2 hcal over ecal seed cluster energy using 2nd hcal depth (using hcal towers within a cone)
      float hcalDepth1OverEcalBc;  // run2 hcal over ecal seed cluster energy using 1st hcal depth (using hcal towers behind clusters)
      float hcalDepth2OverEcalBc;  // run2 hcal over ecal seed cluster energy using 2nd hcal depth (using hcal towers behind clusters)
      std::array<float, 7>
          hcalOverEcal;  // run3 hcal over ecal seed cluster energy per depth (using rechits within a cone)
      std::array<float, 7>
          hcalOverEcalBc;  // run3 hcal over ecal seed cluster energy per depth (using rechits behind clusters)
      std::vector<CaloTowerDetId> hcalTowersBehindClusters;
      bool invalidHcal;  // set to true if the hcal energy estimate is not valid (e.g. the corresponding tower was off or masked)
      bool pre7DepthHcal;  // to work around an ioread rule issue on legacy RECO files
      float sigmaIetaIphi;
      float eMax;
      float e2nd;
      float eTop;
      float eLeft;
      float eRight;
      float eBottom;
      float e2x5Top;
      float e2x5Left;
      float e2x5Right;
      float e2x5Bottom;
      ShowerShape()
          : sigmaEtaEta(std::numeric_limits<float>::max()),
            sigmaIetaIeta(std::numeric_limits<float>::max()),
            sigmaIphiIphi(std::numeric_limits<float>::max()),
            e1x5(0.f),
            e2x5Max(0.f),
            e5x5(0.f),
            r9(-std::numeric_limits<float>::max()),
            hcalDepth1OverEcal(0.f),
            hcalDepth2OverEcal(0.f),
            hcalDepth1OverEcalBc(0.f),
            hcalDepth2OverEcalBc(0.f),
            hcalOverEcal{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            hcalOverEcalBc{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            invalidHcal(false),
            pre7DepthHcal(true),
            sigmaIetaIphi(0.f),
            eMax(0.f),
            e2nd(0.f),
            eTop(0.f),
            eLeft(0.f),
            eRight(0.f),
            eBottom(0.f),
            e2x5Top(0.f),
            e2x5Left(0.f),
            e2x5Right(0.f),
            e2x5Bottom(0.f) {}
    };

    // accessors
    float sigmaEtaEta() const { return showerShape_.sigmaEtaEta; }
    float sigmaIetaIeta() const { return showerShape_.sigmaIetaIeta; }
    float sigmaIphiIphi() const { return showerShape_.sigmaIphiIphi; }
    float e1x5() const { return showerShape_.e1x5; }
    float e2x5Max() const { return showerShape_.e2x5Max; }
    float e5x5() const { return showerShape_.e5x5; }
    float r9() const { return showerShape_.r9; }
    float hcalOverEcal(const ShowerShape &ss, int depth) const {
      if (ss.pre7DepthHcal) {
        if (depth == 0)
          return ss.hcalDepth1OverEcal + ss.hcalDepth2OverEcal;
        else if (depth == 1)
          return ss.hcalDepth1OverEcal;
        else if (depth == 2)
          return ss.hcalDepth2OverEcal;

        return 0.f;
      } else {
        const auto &hovere = ss.hcalOverEcal;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hovere), std::end(hovere), 0.f)
                                            : hovere[depth - 1];
      }
    }
    float hcalOverEcal(int depth = 0) const { return hcalOverEcal(showerShape_, depth); }
    float hcalOverEcalBc(const ShowerShape &ss, int depth) const {
      if (ss.pre7DepthHcal) {
        if (depth == 0)
          return ss.hcalDepth1OverEcalBc + ss.hcalDepth2OverEcalBc;
        else if (depth == 1)
          return ss.hcalDepth1OverEcalBc;
        else if (depth == 2)
          return ss.hcalDepth2OverEcalBc;

        return 0.f;
      } else {
        const auto &hovere = ss.hcalOverEcalBc;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hovere), std::end(hovere), 0.f)
                                            : hovere[depth - 1];
      }
    }
    float hcalOverEcalBc(int depth = 0) const { return hcalOverEcalBc(showerShape_, depth); }
    const std::vector<CaloTowerDetId> &hcalTowersBehindClusters() const {
      return showerShape_.hcalTowersBehindClusters;
    }
    bool hcalOverEcalValid() const { return !showerShape_.invalidHcal; }
    float eLeft() const { return showerShape_.eLeft; }
    float eRight() const { return showerShape_.eRight; }
    float eTop() const { return showerShape_.eTop; }
    float eBottom() const { return showerShape_.eBottom; }
    const ShowerShape &showerShape() const { return showerShape_; }
    // non-zero-suppressed and no-fractions shower shapes
    // ecal energy is always that from the full 5x5
    float full5x5_sigmaEtaEta() const { return full5x5_showerShape_.sigmaEtaEta; }
    float full5x5_sigmaIetaIeta() const { return full5x5_showerShape_.sigmaIetaIeta; }
    float full5x5_sigmaIphiIphi() const { return full5x5_showerShape_.sigmaIphiIphi; }
    float full5x5_e1x5() const { return full5x5_showerShape_.e1x5; }
    float full5x5_e2x5Max() const { return full5x5_showerShape_.e2x5Max; }
    float full5x5_e5x5() const { return full5x5_showerShape_.e5x5; }
    float full5x5_r9() const { return full5x5_showerShape_.r9; }
    float full5x5_hcalOverEcal(int depth = 0) const { return hcalOverEcal(full5x5_showerShape_, depth); }
    float full5x5_hcalOverEcalBc(int depth = 0) const { return hcalOverEcalBc(full5x5_showerShape_, depth); }
    bool full5x5_hcalOverEcalValid() const { return !full5x5_showerShape_.invalidHcal; }
    float full5x5_e2x5Left() const { return full5x5_showerShape_.e2x5Left; }
    float full5x5_e2x5Right() const { return full5x5_showerShape_.e2x5Right; }
    float full5x5_e2x5Top() const { return full5x5_showerShape_.e2x5Top; }
    float full5x5_e2x5Bottom() const { return full5x5_showerShape_.e2x5Bottom; }
    float full5x5_eLeft() const { return full5x5_showerShape_.eLeft; }
    float full5x5_eRight() const { return full5x5_showerShape_.eRight; }
    float full5x5_eTop() const { return full5x5_showerShape_.eTop; }
    float full5x5_eBottom() const { return full5x5_showerShape_.eBottom; }
    const ShowerShape &full5x5_showerShape() const { return full5x5_showerShape_; }

    // setters (if you know what you're doing)
    void setShowerShape(const ShowerShape &s) { showerShape_ = s; }
    void full5x5_setShowerShape(const ShowerShape &s) { full5x5_showerShape_ = s; }

    // for backward compatibility (this will only ever be the ZS shapes!)
    float scSigmaEtaEta() const { return sigmaEtaEta(); }
    float scSigmaIEtaIEta() const { return sigmaIetaIeta(); }
    float scE1x5() const { return e1x5(); }
    float scE2x5Max() const { return e2x5Max(); }
    float scE5x5() const { return e5x5(); }
    float hadronicOverEm() const { return hcalOverEcal(); }

  private:
    // attributes
    ShowerShape showerShape_;
    ShowerShape full5x5_showerShape_;

    //=======================================================
    // SaturationInfo
    //=======================================================

  public:
    struct SaturationInfo {
      int nSaturatedXtals;
      bool isSeedSaturated;
      SaturationInfo() : nSaturatedXtals(0), isSeedSaturated(false){};
    };

    // accessors
    float nSaturatedXtals() const { return saturationInfo_.nSaturatedXtals; }
    float isSeedSaturated() const { return saturationInfo_.isSeedSaturated; }
    const SaturationInfo &saturationInfo() const { return saturationInfo_; }
    void setSaturationInfo(const SaturationInfo &s) { saturationInfo_ = s; }

  private:
    SaturationInfo saturationInfo_;

    //=======================================================
    // Isolation Variables
    //=======================================================

  public:
    struct IsolationVariables {
      float tkSumPt;                           // track iso with electron footprint removed
      float tkSumPtHEEP;                       // track iso used for the HEEP ID
      float ecalRecHitSumEt;                   // ecal iso deposit with electron footprint removed
      float hcalDepth1TowerSumEt;              // hcal depth 1 iso deposit with electron footprint removed
      float hcalDepth2TowerSumEt;              // hcal depth 2 iso deposit with electron footprint removed
      float hcalDepth1TowerSumEtBc;            // hcal depth 1 iso deposit without towers behind clusters
      float hcalDepth2TowerSumEtBc;            // hcal depth 2 iso deposit without towers behind clusters
      std::array<float, 7> hcalRecHitSumEt;    // ...per depth, with electron footprint removed
      std::array<float, 7> hcalRecHitSumEtBc;  // ...per depth, with hcal rechit behind cluster removed
      bool pre7DepthHcal;                      // to work around an ioread rule issue on legacy RECO files
      IsolationVariables()
          : tkSumPt(0.),
            tkSumPtHEEP(0.),
            ecalRecHitSumEt(0.),
            hcalDepth1TowerSumEt(0.f),
            hcalDepth2TowerSumEt(0.f),
            hcalDepth1TowerSumEtBc(0.f),
            hcalDepth2TowerSumEtBc(0.f),
            hcalRecHitSumEt{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            hcalRecHitSumEtBc{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            pre7DepthHcal(true) {}
    };

    // 03 accessors
    float dr03TkSumPt() const { return dr03_.tkSumPt; }
    float dr03TkSumPtHEEP() const { return dr03_.tkSumPtHEEP; }
    float dr03EcalRecHitSumEt() const { return dr03_.ecalRecHitSumEt; }
    float hcalTowerSumEt(const IsolationVariables &iv, int depth) const {
      if (iv.pre7DepthHcal) {
        if (depth == 0)
          return iv.hcalDepth1TowerSumEt + iv.hcalDepth2TowerSumEt;
        else if (depth == 1)
          return iv.hcalDepth1TowerSumEt;
        else if (depth == 2)
          return iv.hcalDepth2TowerSumEt;

        return 0.f;
      } else {
        const auto &hcaliso = iv.hcalRecHitSumEt;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hcaliso), std::end(hcaliso), 0.f)
                                            : hcaliso[depth - 1];
      }
    }
    float dr03HcalTowerSumEt(int depth = 0) const { return hcalTowerSumEt(dr03_, depth); }
    float hcalTowerSumEtBc(const IsolationVariables &iv, int depth) const {
      if (iv.pre7DepthHcal) {
        if (depth == 0)
          return iv.hcalDepth1TowerSumEtBc + iv.hcalDepth2TowerSumEtBc;
        else if (depth == 1)
          return iv.hcalDepth1TowerSumEtBc;
        else if (depth == 2)
          return iv.hcalDepth2TowerSumEtBc;

        return 0.f;
      } else {
        const auto &hcaliso = iv.hcalRecHitSumEtBc;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hcaliso), std::end(hcaliso), 0.f)
                                            : hcaliso[depth - 1];
      }
    }
    float dr03HcalTowerSumEtBc(int depth = 0) const { return hcalTowerSumEtBc(dr03_, depth); }
    const IsolationVariables &dr03IsolationVariables() const { return dr03_; }

    // 04 accessors
    float dr04TkSumPt() const { return dr04_.tkSumPt; }
    float dr04TkSumPtHEEP() const { return dr04_.tkSumPtHEEP; }
    float dr04EcalRecHitSumEt() const { return dr04_.ecalRecHitSumEt; }
    float dr04HcalTowerSumEt(int depth = 0) const { return hcalTowerSumEt(dr04_, depth); }
    float dr04HcalTowerSumEtBc(int depth = 0) const { return hcalTowerSumEtBc(dr04_, depth); }
    const IsolationVariables &dr04IsolationVariables() const { return dr04_; }

    // setters ?!?
    void setDr03Isolation(const IsolationVariables &dr03) { dr03_ = dr03; }
    void setDr04Isolation(const IsolationVariables &dr04) { dr04_ = dr04; }

    // for backward compatibility
    void setIsolation03(const IsolationVariables &dr03) { dr03_ = dr03; }
    void setIsolation04(const IsolationVariables &dr04) { dr04_ = dr04; }
    const IsolationVariables &isolationVariables03() const { return dr03_; }
    const IsolationVariables &isolationVariables04() const { return dr04_; }

    // go back to run2-like 2 effective depths if desired - depth 1 is the normal depth 1, depth 2 is the sum over the rest
    void hcalToRun2EffDepth();

  private:
    // attributes
    IsolationVariables dr03_;
    IsolationVariables dr04_;

    //=======================================================
    // Conversion Rejection Information
    //=======================================================

  public:
    struct ConversionRejection {
      int flags;             // -max:not-computed, other: as computed by Puneeth conversion code
      TrackBaseRef partner;  // conversion partner
      float dist;            // distance to the conversion partner
      float dcot;            // difference of cot(angle) with the conversion partner track
      float radius;          // signed conversion radius
      float vtxFitProb;      //fit probablity (chi2/ndof) of the matched conversion vtx
      ConversionRejection()
          : flags(-1),
            dist(std::numeric_limits<float>::max()),
            dcot(std::numeric_limits<float>::max()),
            radius(std::numeric_limits<float>::max()),
            vtxFitProb(std::numeric_limits<float>::max()) {}
    };

    // accessors
    int convFlags() const { return conversionRejection_.flags; }
    TrackBaseRef convPartner() const { return conversionRejection_.partner; }
    float convDist() const { return conversionRejection_.dist; }
    float convDcot() const { return conversionRejection_.dcot; }
    float convRadius() const { return conversionRejection_.radius; }
    float convVtxFitProb() const { return conversionRejection_.vtxFitProb; }
    const ConversionRejection &conversionRejectionVariables() const { return conversionRejection_; }
    void setConversionRejectionVariables(const ConversionRejection &convRej) { conversionRejection_ = convRej; }

  private:
    // attributes
    ConversionRejection conversionRejection_;

    //=======================================================
    // Pflow Information
    //=======================================================

  public:
    struct PflowIsolationVariables {
      //first three data members that changed names, according to DataFormats/MuonReco/interface/MuonPFIsolation.h
      float sumChargedHadronPt;  //!< sum-pt of charged Hadron    // old float chargedHadronIso ;
      float sumNeutralHadronEt;  //!< sum pt of neutral hadrons  // old float neutralHadronIso ;
      float sumPhotonEt;         //!< sum pt of PF photons              // old float photonIso ;
      //then four new data members, corresponding to DataFormats/MuonReco/interface/MuonPFIsolation.h
      float sumChargedParticlePt;             //!< sum-pt of charged Particles(inludes e/mu)
      float sumNeutralHadronEtHighThreshold;  //!< sum pt of neutral hadrons with a higher threshold
      float sumPhotonEtHighThreshold;         //!< sum pt of PF photons with a higher threshold
      float sumPUPt;                          //!< sum pt of charged Particles not from PV  (for Pu corrections)
      //new pf cluster based isolation values
      float sumEcalClusterEt;  //sum pt of ecal clusters, vetoing clusters part of electron
      float sumHcalClusterEt;  //sum pt of hcal clusters, vetoing clusters part of electron
      PflowIsolationVariables()
          : sumChargedHadronPt(0),
            sumNeutralHadronEt(0),
            sumPhotonEt(0),
            sumChargedParticlePt(0),
            sumNeutralHadronEtHighThreshold(0),
            sumPhotonEtHighThreshold(0),
            sumPUPt(0),
            sumEcalClusterEt(0),
            sumHcalClusterEt(0){};
    };

    struct MvaInput {
      int earlyBrem;                // Early Brem detected (-2=>unknown,-1=>could not be evaluated,0=>wrong,1=>true)
      int lateBrem;                 // Late Brem detected (-2=>unknown,-1=>could not be evaluated,0=>wrong,1=>true)
      float sigmaEtaEta;            // Sigma-eta-eta with the PF cluster
      float hadEnergy;              // Associated PF Had Cluster energy
      float deltaEta;               // PF-cluster GSF track delta-eta
      int nClusterOutsideMustache;  // -2 => unknown, -1 =>could not be evaluated, 0 and more => number of clusters
      float etOutsideMustache;
      MvaInput()
          : earlyBrem(-2),
            lateBrem(-2),
            sigmaEtaEta(std::numeric_limits<float>::max()),
            hadEnergy(0.),
            deltaEta(std::numeric_limits<float>::max()),
            nClusterOutsideMustache(-2),
            etOutsideMustache(-std::numeric_limits<float>::max()) {}
    };

    static constexpr float mvaPlaceholder = -999999999.;

    struct MvaOutput {
      int status;  // see PFCandidateElectronExtra::StatusFlag
      float mva_Isolated;
      float mva_e_pi;
      float mvaByPassForIsolated;  // complementary MVA used in preselection
      float dnn_e_sigIsolated;
      float dnn_e_sigNonIsolated;
      float dnn_e_bkgNonIsolated;
      float dnn_e_bkgTau;
      float dnn_e_bkgPhoton;
      MvaOutput()
          : status(-1),
            mva_Isolated(mvaPlaceholder),
            mva_e_pi(mvaPlaceholder),
            mvaByPassForIsolated(mvaPlaceholder),
            dnn_e_sigIsolated(mvaPlaceholder),
            dnn_e_sigNonIsolated(mvaPlaceholder),
            dnn_e_bkgNonIsolated(mvaPlaceholder),
            dnn_e_bkgTau(mvaPlaceholder),
            dnn_e_bkgPhoton(mvaPlaceholder) {}
    };

    // accessors
    const PflowIsolationVariables &pfIsolationVariables() const { return pfIso_; }
    //backwards compat functions for pat::Electron
    float ecalPFClusterIso() const { return pfIso_.sumEcalClusterEt; };
    float hcalPFClusterIso() const { return pfIso_.sumHcalClusterEt; };

    const MvaInput &mvaInput() const { return mvaInput_; }
    const MvaOutput &mvaOutput() const { return mvaOutput_; }

    // setters
    void setPfIsolationVariables(const PflowIsolationVariables &iso) { pfIso_ = iso; }
    void setMvaInput(const MvaInput &mi) { mvaInput_ = mi; }
    void setMvaOutput(const MvaOutput &mo) { mvaOutput_ = mo; }

    // for backward compatibility
    float mva_Isolated() const { return mvaOutput_.mva_Isolated; }
    float mva_e_pi() const { return mvaOutput_.mva_e_pi; }
    float dnn_signal_Isolated() const { return mvaOutput_.dnn_e_sigIsolated; }
    float dnn_signal_nonIsolated() const { return mvaOutput_.dnn_e_sigNonIsolated; }
    float dnn_bkg_nonIsolated() const { return mvaOutput_.dnn_e_bkgNonIsolated; }
    float dnn_bkg_Tau() const { return mvaOutput_.dnn_e_bkgTau; }
    float dnn_bkg_Photon() const { return mvaOutput_.dnn_e_bkgPhoton; }

  private:
    PflowIsolationVariables pfIso_;
    MvaInput mvaInput_;
    MvaOutput mvaOutput_;

    //=======================================================
    // Preselection and Ambiguity
    //=======================================================

  public:
    // accessors
    bool ecalDriven() const;  // return true if ecalDrivenSeed() and passingCutBasedPreselection()
    bool passingCutBasedPreselection() const { return passCutBasedPreselection_; }
    bool passingPflowPreselection() const { return passPflowPreselection_; }
    bool ambiguous() const { return ambiguous_; }
    GsfTrackRefVector::size_type ambiguousGsfTracksSize() const { return ambiguousGsfTracks_.size(); }
    auto const &ambiguousGsfTracks() const { return ambiguousGsfTracks_; }

    // setters
    void setPassCutBasedPreselection(bool flag) { passCutBasedPreselection_ = flag; }
    void setPassPflowPreselection(bool flag) { passPflowPreselection_ = flag; }
    void setAmbiguous(bool flag) { ambiguous_ = flag; }
    void clearAmbiguousGsfTracks() { ambiguousGsfTracks_.clear(); }
    void addAmbiguousGsfTrack(const reco::GsfTrackRef &t) { ambiguousGsfTracks_.push_back(t); }

    // backward compatibility
    void setPassMvaPreselection(bool flag) { passMvaPreslection_ = flag; }
    bool passingMvaPreselection() const { return passMvaPreslection_; }

  private:
    // attributes
    bool passCutBasedPreselection_;
    bool passPflowPreselection_;
    bool passMvaPreslection_;  // to be removed : passPflowPreslection_
    bool ambiguous_;
    GsfTrackRefVector ambiguousGsfTracks_;  // ambiguous gsf tracks

    //=======================================================
    // Brem Fractions and Classification
    //=======================================================

  public:
    struct ClassificationVariables {
      float trackFbrem;  // the brem fraction from gsf fit: (track momentum in - track momentum out) / track momentum in
      float superClusterFbrem;  // the brem fraction from supercluster: (supercluster energy - electron cluster energy) / supercluster energy
      constexpr static float kDefaultValue = -1.e30;
      ClassificationVariables() : trackFbrem(kDefaultValue), superClusterFbrem(kDefaultValue) {}
    };
    enum Classification { UNKNOWN = -1, GOLDEN = 0, BIGBREM = 1, BADTRACK = 2, SHOWERING = 3, GAP = 4 };

    // accessors
    float trackFbrem() const { return classVariables_.trackFbrem; }
    float superClusterFbrem() const { return classVariables_.superClusterFbrem; }
    const ClassificationVariables &classificationVariables() const { return classVariables_; }
    Classification classification() const { return class_; }

    // utilities
    int numberOfBrems() const { return basicClustersSize() - 1; }
    float fbrem() const { return trackFbrem(); }

    // setters
    void setTrackFbrem(float fbrem) { classVariables_.trackFbrem = fbrem; }
    void setSuperClusterFbrem(float fbrem) { classVariables_.superClusterFbrem = fbrem; }
    void setClassificationVariables(const ClassificationVariables &cv) { classVariables_ = cv; }
    void setClassification(Classification myclass) { class_ = myclass; }

  private:
    // attributes
    ClassificationVariables classVariables_;
    Classification class_;  // fbrem and number of clusters based electron classification

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
    // hcalOverEcal, eSuperClusterOverP,
    // eSeedClusterOverP, eEleClusterOverPout.
    //=======================================================

  public:
    enum P4Kind { P4_UNKNOWN = -1, P4_FROM_SUPER_CLUSTER = 0, P4_COMBINATION = 1, P4_PFLOW_COMBINATION = 2 };

    struct Corrections {
      bool isEcalEnergyCorrected;  // true if ecal energy has been corrected
      float correctedEcalEnergy;  // corrected energy (if !isEcalEnergyCorrected this value is identical to the supercluster energy)
      float correctedEcalEnergyError;  // error on energy
      //bool isMomentumCorrected ;     // DEPRECATED
      float trackMomentumError;  // track momentum error from gsf fit
      //
      LorentzVector fromSuperClusterP4;  // for P4_FROM_SUPER_CLUSTER
      float fromSuperClusterP4Error;     // for P4_FROM_SUPER_CLUSTER
      LorentzVector combinedP4;          // for P4_COMBINATION
      float combinedP4Error;             // for P4_COMBINATION
      LorentzVector pflowP4;             // for P4_PFLOW_COMBINATION
      float pflowP4Error;                // for P4_PFLOW_COMBINATION
      P4Kind candidateP4Kind;            // say which momentum has been stored in reco::Candidate
      //
      Corrections()
          : isEcalEnergyCorrected(false),
            correctedEcalEnergy(0.),
            correctedEcalEnergyError(999.),
            /*isMomentumCorrected(false),*/ trackMomentumError(999.),
            fromSuperClusterP4Error(999.),
            combinedP4Error(999.),
            pflowP4Error(999.),
            candidateP4Kind(P4_UNKNOWN) {}
    };

    // setters
    void setCorrectedEcalEnergyError(float newEnergyError);
    void setCorrectedEcalEnergy(float newEnergy);
    void setCorrectedEcalEnergy(float newEnergy, bool rescaleDependentValues);
    void setTrackMomentumError(float trackMomentumError);
    void setP4(P4Kind kind, const LorentzVector &p4, float p4Error, bool setCandidate);
    using RecoCandidate::setP4;

    // accessors
    bool isEcalEnergyCorrected() const { return corrections_.isEcalEnergyCorrected; }
    float correctedEcalEnergy() const { return corrections_.correctedEcalEnergy; }
    float correctedEcalEnergyError() const { return corrections_.correctedEcalEnergyError; }
    float trackMomentumError() const { return corrections_.trackMomentumError; }
    const LorentzVector &p4(P4Kind kind) const;
    using RecoCandidate::p4;
    float p4Error(P4Kind kind) const;
    P4Kind candidateP4Kind() const { return corrections_.candidateP4Kind; }
    const Corrections &corrections() const { return corrections_; }

    // bare setter (if you know what you're doing)
    void setCorrections(const Corrections &c) { corrections_ = c; }

    // for backward compatibility
    void setEcalEnergyError(float energyError) { setCorrectedEcalEnergyError(energyError); }
    float ecalEnergy() const { return correctedEcalEnergy(); }
    float ecalEnergyError() const { return correctedEcalEnergyError(); }
    //bool isMomentumCorrected() const { return corrections_.isMomentumCorrected ; }
    float caloEnergy() const { return correctedEcalEnergy(); }
    bool isEnergyScaleCorrected() const { return isEcalEnergyCorrected(); }
    void correctEcalEnergy(float newEnergy, float newEnergyError, bool corrEovP = true) {
      setCorrectedEcalEnergy(newEnergy, corrEovP);
      setEcalEnergyError(newEnergyError);
    }
    void correctMomentum(const LorentzVector &p4, float trackMomentumError, float p4Error) {
      setTrackMomentumError(trackMomentumError);
      setP4(P4_COMBINATION, p4, p4Error, true);
    }

  private:
    // attributes
    Corrections corrections_;

  public:
    struct PixelMatchVariables {
      //! Pixel match variable: deltaPhi for innermost hit
      float dPhi1;
      //! Pixel match variable: deltaPhi for second hit
      float dPhi2;
      //! Pixel match variable: deltaRz for innermost hit
      float dRz1;
      //! Pixel match variable: deltaRz for second hit
      float dRz2;
      //! Subdetectors for first and second pixel hit
      unsigned char subdetectors;
      PixelMatchVariables() : dPhi1(-999), dPhi2(-999), dRz1(-999), dRz2(-999), subdetectors(0) {}
      ~PixelMatchVariables() {}
    };
    void setPixelMatchSubdetectors(int sd1, int sd2) { pixelMatchVariables_.subdetectors = 10 * sd1 + sd2; }
    void setPixelMatchDPhi1(float dPhi1) { pixelMatchVariables_.dPhi1 = dPhi1; }
    void setPixelMatchDPhi2(float dPhi2) { pixelMatchVariables_.dPhi2 = dPhi2; }
    void setPixelMatchDRz1(float dRz1) { pixelMatchVariables_.dRz1 = dRz1; }
    void setPixelMatchDRz2(float dRz2) { pixelMatchVariables_.dRz2 = dRz2; }

    int pixelMatchSubdetector1() const { return pixelMatchVariables_.subdetectors / 10; }
    int pixelMatchSubdetector2() const { return pixelMatchVariables_.subdetectors % 10; }
    float pixelMatchDPhi1() const { return pixelMatchVariables_.dPhi1; }
    float pixelMatchDPhi2() const { return pixelMatchVariables_.dPhi2; }
    float pixelMatchDRz1() const { return pixelMatchVariables_.dRz1; }
    float pixelMatchDRz2() const { return pixelMatchVariables_.dRz2; }

  private:
    PixelMatchVariables pixelMatchVariables_;
  };

}  // namespace reco

#endif
