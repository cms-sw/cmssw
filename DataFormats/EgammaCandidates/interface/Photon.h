#ifndef EgammaCandidates_Photon_h
#define EgammaCandidates_Photon_h
/** \class reco::Photon 
 *
 * \author  N. Marinelli Univ. of Notre Dame
 * Photon object built out of PhotonCore
 * stores isolation, shower shape and additional info
 * needed for identification
 * 
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include <numeric>

namespace reco {

  class Photon : public RecoCandidate {
  public:
    /// Forward declaration of data structures included in the object
    struct FiducialFlags;
    struct IsolationVariables;
    struct ShowerShape;
    struct MIPVariables;
    struct SaturationInfo;

    /// default constructor
    Photon() : RecoCandidate() { pixelSeed_ = false; }

    /// copy constructor
    Photon(const Photon&);

    /// constructor from values
    Photon(const LorentzVector& p4, const Point& caloPos, const PhotonCoreRef& core, const Point& vtx = Point(0, 0, 0));

    /// destructor
    ~Photon() override;

    /// returns a clone of the candidate
    Photon* clone() const override;

    /// returns a reference to the core photon object
    reco::PhotonCoreRef photonCore() const { return photonCore_; }
    void setPhotonCore(const reco::PhotonCoreRef& photonCore) { photonCore_ = photonCore; }

    //
    /// Retrieve photonCore attributes
    //
    // retrieve provenance
    bool isPFlowPhoton() const { return this->photonCore()->isPFlowPhoton(); }
    bool isStandardPhoton() const { return this->photonCore()->isStandardPhoton(); }
    /// Ref to SuperCluster
    reco::SuperClusterRef superCluster() const override;
    /// Ref to PFlow SuperCluster
    reco::SuperClusterRef parentSuperCluster() const { return this->photonCore()->parentSuperCluster(); }
    /// vector of references to  Conversion's
    reco::ConversionRefVector conversions() const { return this->photonCore()->conversions(); }
    enum ConversionProvenance { egamma = 0, pflow = 1, both = 2 };

    /// vector of references to  one leg Conversion's
    reco::ConversionRefVector conversionsOneLeg() const { return this->photonCore()->conversionsOneLeg(); }
    /// Bool flagging photons with a vector of refereces to conversions with size >0
    bool hasConversionTracks() const {
      if (!this->photonCore()->conversions().empty() || !this->photonCore()->conversionsOneLeg().empty())
        return true;
      else
        return false;
    }
    /// reference to electron Pixel seed
    reco::ElectronSeedRefVector electronPixelSeeds() const { return this->photonCore()->electronPixelSeeds(); }
    /// Bool flagging photons having a non-zero size vector of Ref to electornPixel seeds
    bool hasPixelSeed() const {
      if (!(this->photonCore()->electronPixelSeeds()).empty())
        return true;
      else
        return false;
    }
    int conversionTrackProvenance(const edm::RefToBase<reco::Track>& convTrack) const;

    /// position in ECAL: this is th SC position if r9<0.93. If r8>0.93 is position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPointF caloPosition() const { return caloPosition_; }
    /// set primary event vertex used to define photon direction
    void setVertex(const Point& vertex) override;
    /// Implement Candidate method for particle species
    bool isPhoton() const override { return true; }

    //=======================================================
    // Fiducial Flags
    //=======================================================
    struct FiducialFlags {
      //Fiducial flags
      bool isEB;         //Photon is in EB
      bool isEE;         //Photon is in EE
      bool isEBEtaGap;   //Photon is in supermodule/supercrystal eta gap in EB
      bool isEBPhiGap;   //Photon is in supermodule/supercrystal phi gap in EB
      bool isEERingGap;  //Photon is in crystal ring gap in EE
      bool isEEDeeGap;   //Photon is in crystal dee gap in EE
      bool isEBEEGap;    //Photon is in border between EB and EE.

      FiducialFlags()
          : isEB(false),
            isEE(false),
            isEBEtaGap(false),
            isEBPhiGap(false),
            isEERingGap(false),
            isEEDeeGap(false),
            isEBEEGap(false)

      {}
    };

    /// set flags for photons in the ECAL fiducial volume
    void setFiducialVolumeFlags(const FiducialFlags& a) { fiducialFlagBlock_ = a; }
    /// Ritrievs fiducial flags
    /// true if photon is in ECAL barrel
    bool isEB() const { return fiducialFlagBlock_.isEB; }
    // true if photon is in ECAL endcap
    bool isEE() const { return fiducialFlagBlock_.isEE; }
    /// true if photon is in EB, and inside the boundaries in super crystals/modules
    bool isEBGap() const { return (isEBEtaGap() || isEBPhiGap()); }
    bool isEBEtaGap() const { return fiducialFlagBlock_.isEBEtaGap; }
    bool isEBPhiGap() const { return fiducialFlagBlock_.isEBPhiGap; }
    /// true if photon is in EE, and inside the boundaries in supercrystal/D
    bool isEEGap() const { return (isEERingGap() || isEEDeeGap()); }
    bool isEERingGap() const { return fiducialFlagBlock_.isEERingGap; }
    bool isEEDeeGap() const { return fiducialFlagBlock_.isEEDeeGap; }
    /// true if photon is in boundary between EB and EE
    bool isEBEEGap() const { return fiducialFlagBlock_.isEBEEGap; }

    //=======================================================
    // Shower Shape Variables
    //=======================================================

    struct ShowerShape {
      float sigmaEtaEta;
      float sigmaIetaIeta;
      float e1x5;
      float e2x5;
      float e3x3;
      float e5x5;
      float maxEnergyXtal;
      float hcalDepth1OverEcal;  // hcal over ecal energy using first hcal depth
      float hcalDepth2OverEcal;  // hcal over ecal energy using 2nd hcal depth
      float hcalDepth1OverEcalBc;
      float hcalDepth2OverEcalBc;
      std::array<float, 7> hcalOverEcal;  // hcal over ecal seed cluster energy per depth (using rechits within a cone)
      std::array<float, 7>
          hcalOverEcalBc;  // hcal over ecal seed cluster energy per depth (using rechits behind clusters)
      std::vector<CaloTowerDetId> hcalTowersBehindClusters;
      bool invalidHcal;
      bool pre7DepthHcal;  // to work around an ioread rule issue on legacy RECO files
      float effSigmaRR;
      float sigmaIetaIphi;
      float sigmaIphiIphi;
      float e2nd;
      float eTop;
      float eLeft;
      float eRight;
      float eBottom;
      float e1x3;
      float e2x2;
      float e2x5Max;
      float e2x5Left;
      float e2x5Right;
      float e2x5Top;
      float e2x5Bottom;
      float smMajor;
      float smMinor;
      float smAlpha;
      ShowerShape()
          : sigmaEtaEta(std::numeric_limits<float>::max()),
            sigmaIetaIeta(std::numeric_limits<float>::max()),
            e1x5(0.f),
            e2x5(0.f),
            e3x3(0.f),
            e5x5(0.f),
            maxEnergyXtal(0.f),
            hcalDepth1OverEcal(0.f),
            hcalDepth2OverEcal(0.f),
            hcalDepth1OverEcalBc(0.f),
            hcalDepth2OverEcalBc(0.f),
            hcalOverEcal{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            hcalOverEcalBc{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            invalidHcal(false),
            pre7DepthHcal(true),
            effSigmaRR(std::numeric_limits<float>::max()),
            sigmaIetaIphi(std::numeric_limits<float>::max()),
            sigmaIphiIphi(std::numeric_limits<float>::max()),
            e2nd(0.f),
            eTop(0.f),
            eLeft(0.f),
            eRight(0.f),
            eBottom(0.f),
            e1x3(0.f),
            e2x2(0.f),
            e2x5Max(0.f),
            e2x5Left(0.f),
            e2x5Right(0.f),
            e2x5Top(0.f),
            e2x5Bottom(0.f),
            smMajor(0.f),
            smMinor(0.f),
            smAlpha(0.f) {}
    };
    const ShowerShape& showerShapeVariables() const { return showerShapeBlock_; }
    const ShowerShape& full5x5_showerShapeVariables() const { return full5x5_showerShapeBlock_; }

    void setShowerShapeVariables(const ShowerShape& a) { showerShapeBlock_ = a; }
    void full5x5_setShowerShapeVariables(const ShowerShape& a) { full5x5_showerShapeBlock_ = a; }

    /// the total hadronic over electromagnetic fraction
    float hcalOverEcal(const ShowerShape& ss, int depth) const {
      if (ss.pre7DepthHcal) {
        if (depth == 0)
          return ss.hcalDepth1OverEcal + ss.hcalDepth2OverEcal;
        else if (depth == 1)
          return ss.hcalDepth1OverEcal;
        else if (depth == 2)
          return ss.hcalDepth2OverEcal;

        return 0.f;
      } else {
        const auto& hovere = ss.hcalOverEcal;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hovere), std::end(hovere), 0.f)
                                            : hovere[depth - 1];
      }
    }
    float hcalOverEcal(int depth = 0) const { return hcalOverEcal(showerShapeBlock_, depth); }
    float hadronicOverEm(int depth = 0) const { return hcalOverEcal(depth); }

    /// the ratio of total energy of hcal rechits behind the SC and the SC energy
    float hcalOverEcalBc(const ShowerShape& ss, int depth) const {
      if (ss.pre7DepthHcal) {
        if (depth == 0)
          return ss.hcalDepth1OverEcalBc + ss.hcalDepth2OverEcalBc;
        else if (depth == 1)
          return ss.hcalDepth1OverEcalBc;
        else if (depth == 2)
          return ss.hcalDepth2OverEcalBc;

        return 0.f;
      } else {
        const auto& hovere = ss.hcalOverEcalBc;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hovere), std::end(hovere), 0.f)
                                            : hovere[depth - 1];
      }
    }
    float hcalOverEcalBc(int depth = 0) const { return hcalOverEcalBc(showerShapeBlock_, depth); }
    float hadTowOverEm(int depth = 0) const { return hcalOverEcalBc(depth); }

    const std::vector<CaloTowerDetId>& hcalTowersBehindClusters() const {
      return showerShapeBlock_.hcalTowersBehindClusters;
    }

    /// returns false if H/E is not reliably estimated (e.g. because hcal was off or masked)
    bool hadronicOverEmValid() const { return !showerShapeBlock_.invalidHcal; }
    bool hadTowOverEmValid() const { return !showerShapeBlock_.invalidHcal; }

    ///  Shower shape variables
    float e1x5() const { return showerShapeBlock_.e1x5; }
    float e2x5() const { return showerShapeBlock_.e2x5; }
    float e3x3() const { return showerShapeBlock_.e3x3; }
    float e5x5() const { return showerShapeBlock_.e5x5; }
    float maxEnergyXtal() const { return showerShapeBlock_.maxEnergyXtal; }
    float sigmaEtaEta() const { return showerShapeBlock_.sigmaEtaEta; }
    float sigmaIetaIeta() const { return showerShapeBlock_.sigmaIetaIeta; }
    float r1x5() const { return showerShapeBlock_.e1x5 / showerShapeBlock_.e5x5; }
    float r2x5() const { return showerShapeBlock_.e2x5 / showerShapeBlock_.e5x5; }
    float r9() const { return showerShapeBlock_.e3x3 / this->superCluster()->rawEnergy(); }

    ///full5x5 Shower shape variables
    float full5x5_e1x5() const { return full5x5_showerShapeBlock_.e1x5; }
    float full5x5_e2x5() const { return full5x5_showerShapeBlock_.e2x5; }
    float full5x5_e3x3() const { return full5x5_showerShapeBlock_.e3x3; }
    float full5x5_e5x5() const { return full5x5_showerShapeBlock_.e5x5; }
    float full5x5_maxEnergyXtal() const { return full5x5_showerShapeBlock_.maxEnergyXtal; }
    float full5x5_sigmaEtaEta() const { return full5x5_showerShapeBlock_.sigmaEtaEta; }
    float full5x5_sigmaIetaIeta() const { return full5x5_showerShapeBlock_.sigmaIetaIeta; }
    float full5x5_r1x5() const { return full5x5_showerShapeBlock_.e1x5 / full5x5_showerShapeBlock_.e5x5; }
    float full5x5_r2x5() const { return full5x5_showerShapeBlock_.e2x5 / full5x5_showerShapeBlock_.e5x5; }
    float full5x5_r9() const { return full5x5_showerShapeBlock_.e3x3 / this->superCluster()->rawEnergy(); }

    /// the total hadronic over electromagnetic fraction
    float full5x5_hcalOverEcal(int depth = 0) const { return hcalOverEcal(full5x5_showerShapeBlock_, depth); }
    float full5x5_hadronicOverEm(int depth = 0) const { return full5x5_hcalOverEcal(depth); }

    /// the ratio of total energy of hcal rechits behind the SC and the SC energy
    float full5x5_hcalOverEcalBc(int depth = 0) const { return hcalOverEcalBc(full5x5_showerShapeBlock_, depth); }
    float full5x5_hadTowOverEm(int depth = 0) const { return full5x5_hcalOverEcalBc(depth); }

    //=======================================================
    // SaturationInfo
    //=======================================================

    struct SaturationInfo {
      int nSaturatedXtals;
      bool isSeedSaturated;
      SaturationInfo() : nSaturatedXtals(0), isSeedSaturated(false){};
    };

    // accessors
    float nSaturatedXtals() const { return saturationInfo_.nSaturatedXtals; }
    float isSeedSaturated() const { return saturationInfo_.isSeedSaturated; }
    const SaturationInfo& saturationInfo() const { return saturationInfo_; }
    void setSaturationInfo(const SaturationInfo& s) { saturationInfo_ = s; }

    //=======================================================
    // Energy Determinations
    //=======================================================
    enum P4type { undefined = -1, ecal_standard = 0, ecal_photons = 1, regression1 = 2, regression2 = 3 };

    struct EnergyCorrections {
      float scEcalEnergy;
      float scEcalEnergyError;
      LorentzVector scEcalP4;
      float phoEcalEnergy;
      float phoEcalEnergyError;
      LorentzVector phoEcalP4;
      float regression1Energy;
      float regression1EnergyError;
      LorentzVector regression1P4;
      float regression2Energy;
      float regression2EnergyError;
      LorentzVector regression2P4;
      P4type candidateP4type;
      EnergyCorrections()
          : scEcalEnergy(0.),
            scEcalEnergyError(999.),
            scEcalP4(0., 0., 0., 0.),
            phoEcalEnergy(0.),
            phoEcalEnergyError(999.),
            phoEcalP4(0., 0., 0., 0.),
            regression1Energy(0.),
            regression1EnergyError(999.),
            regression1P4(0., 0., 0., 0.),
            regression2Energy(0.),
            regression2EnergyError(999.),
            regression2P4(0., 0., 0., 0.),
            candidateP4type(undefined) {}
    };

    using RecoCandidate::p4;
    using RecoCandidate::setP4;

    //sets both energy and its uncertainty
    void setCorrectedEnergy(P4type type, float E, float dE, bool toCand = true);
    void setP4(P4type type, const LorentzVector& p4, float p4Error, bool setToRecoCandidate);
    void setEnergyCorrections(const EnergyCorrections& e) { eCorrections_ = e; }
    void setCandidateP4type(const P4type type) { eCorrections_.candidateP4type = type; }

    float getCorrectedEnergy(P4type type) const;
    float getCorrectedEnergyError(P4type type) const;
    P4type getCandidateP4type() const { return eCorrections_.candidateP4type; }
    const LorentzVector& p4(P4type type) const;
    const EnergyCorrections& energyCorrections() const { return eCorrections_; }

    //=======================================================
    // MIP Variables
    //=======================================================

    struct MIPVariables {
      float mipChi2;
      float mipTotEnergy;
      float mipSlope;
      float mipIntercept;
      int mipNhitCone;
      bool mipIsHalo;

      MIPVariables()
          :

            mipChi2(0),
            mipTotEnergy(0),
            mipSlope(0),
            mipIntercept(0),
            mipNhitCone(0),
            mipIsHalo(false) {}
    };

    ///  MIP variables
    float mipChi2() const { return mipVariableBlock_.mipChi2; }
    float mipTotEnergy() const { return mipVariableBlock_.mipTotEnergy; }
    float mipSlope() const { return mipVariableBlock_.mipSlope; }
    float mipIntercept() const { return mipVariableBlock_.mipIntercept; }
    int mipNhitCone() const { return mipVariableBlock_.mipNhitCone; }
    bool mipIsHalo() const { return mipVariableBlock_.mipIsHalo; }

    ///set mip Variables
    void setMIPVariables(const MIPVariables& mipVar) { mipVariableBlock_ = mipVar; }

    //=======================================================
    // Isolation Variables
    //=======================================================

    struct IsolationVariables {
      //These are analysis quantities calculated in the PhotonIDAlgo class

      //EcalRecHit isolation
      float ecalRecHitSumEt;
      //HcalTower isolation
      float hcalTowerSumEt;
      //HcalDepth1Tower isolation
      float hcalDepth1TowerSumEt;
      //HcalDepth2Tower isolation
      float hcalDepth2TowerSumEt;
      //HcalTower isolation subtracting the hadronic energy in towers behind the BCs in the SC
      float hcalTowerSumEtBc;
      //HcalDepth1Tower isolation subtracting the hadronic energy in towers behind the BCs in the SC
      float hcalDepth1TowerSumEtBc;
      //HcalDepth2Tower isolation subtracting the hadronic energy in towers behind the BCs in the SC
      float hcalDepth2TowerSumEtBc;
      std::array<float, 7> hcalRecHitSumEt;    // ...per depth, with photon footprint within a cone removed
      std::array<float, 7> hcalRecHitSumEtBc;  // ...per depth, with hcal rechits behind cluster removed
      bool pre7DepthHcal;                      // to work around an ioread rule issue on legacy RECO files
      //Sum of track pT in a cone of dR
      float trkSumPtSolidCone;
      //Sum of track pT in a hollow cone of outer radius, inner radius
      float trkSumPtHollowCone;
      //Number of tracks in a cone of dR
      int nTrkSolidCone;
      //Number of tracks in a hollow cone of outer radius, inner radius
      int nTrkHollowCone;
      IsolationVariables()
          :

            ecalRecHitSumEt(0.f),
            hcalTowerSumEt(0.f),
            hcalDepth1TowerSumEt(0.f),
            hcalDepth2TowerSumEt(0.f),
            hcalTowerSumEtBc(0.f),
            hcalDepth1TowerSumEtBc(0.f),
            hcalDepth2TowerSumEtBc(0.f),
            hcalRecHitSumEt{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            hcalRecHitSumEtBc{{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}},
            pre7DepthHcal(true),
            trkSumPtSolidCone(0.f),
            trkSumPtHollowCone(0.f),
            nTrkSolidCone(0),
            nTrkHollowCone(0) {}
    };

    /// set relevant isolation variables
    void setIsolationVariables(const IsolationVariables& isolInDr04, const IsolationVariables& isolInDr03) {
      isolationR04_ = isolInDr04;
      isolationR03_ = isolInDr03;
    }

    /// Egamma Isolation variables in cone dR=0.4
    ///Ecal isolation sum calculated from recHits
    float ecalRecHitSumEtConeDR04() const { return isolationR04_.ecalRecHitSumEt; }
    /// Hcal isolation sum for each depth excluding the region containing the rechits used for hcalOverEcal()
    float hcalTowerSumEt(const IsolationVariables& iv, int depth) const {
      if (iv.pre7DepthHcal) {
        if (depth == 0)
          return iv.hcalTowerSumEt;
        else if (depth == 1)
          return iv.hcalDepth1TowerSumEt;
        else if (depth == 2)
          return iv.hcalDepth2TowerSumEt;

        return 0.f;
      } else {
        const auto& hcaliso = iv.hcalRecHitSumEt;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hcaliso), std::end(hcaliso), 0.f)
                                            : hcaliso[depth - 1];
      }
    }
    float hcalTowerSumEtConeDR04(int depth = 0) const { return hcalTowerSumEt(isolationR04_, depth); }
    /// Hcal isolation sum for each depth excluding the region containing the rechits used for hcalOverEcalBc()
    float hcalTowerSumEtBc(const IsolationVariables& iv, int depth) const {
      if (iv.pre7DepthHcal) {
        if (depth == 0)
          return iv.hcalTowerSumEtBc;
        else if (depth == 1)
          return iv.hcalDepth1TowerSumEtBc;
        else if (depth == 2)
          return iv.hcalDepth2TowerSumEtBc;

        return 0.f;
      } else {
        const auto& hcaliso = iv.hcalRecHitSumEtBc;
        return (!(depth > 0 and depth < 8)) ? std::accumulate(std::begin(hcaliso), std::end(hcaliso), 0.f)
                                            : hcaliso[depth - 1];
      }
    }
    float hcalTowerSumEtBcConeDR04(int depth = 0) const { return hcalTowerSumEtBc(isolationR04_, depth); }
    //  Track pT sum
    float trkSumPtSolidConeDR04() const { return isolationR04_.trkSumPtSolidCone; }
    //As above, excluding the core at the center of the cone
    float trkSumPtHollowConeDR04() const { return isolationR04_.trkSumPtHollowCone; }
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR04() const { return isolationR04_.nTrkSolidCone; }
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR04() const { return isolationR04_.nTrkHollowCone; }
    //
    /// Isolation variables in cone dR=0.3
    float ecalRecHitSumEtConeDR03() const { return isolationR03_.ecalRecHitSumEt; }
    /// Hcal isolation sum for each depth excluding the region containing the rechits used for hcalOverEcal()
    float hcalTowerSumEtConeDR03(int depth = 0) const { return hcalTowerSumEt(isolationR03_, depth); }
    /// Hcal isolation sum for each depth excluding the region containing the rechits used for hcalOverEcalBc()
    float hcalTowerSumEtBcConeDR03(int depth = 0) const { return hcalTowerSumEtBc(isolationR03_, depth); }
    //  Track pT sum c
    float trkSumPtSolidConeDR03() const { return isolationR03_.trkSumPtSolidCone; }
    //As above, excluding the core at the center of the cone
    float trkSumPtHollowConeDR03() const { return isolationR03_.trkSumPtHollowCone; }
    //Returns number of tracks in a cone of dR
    int nTrkSolidConeDR03() const { return isolationR03_.nTrkSolidCone; }
    //As above, excluding the core at the center of the cone
    int nTrkHollowConeDR03() const { return isolationR03_.nTrkHollowCone; }

    //=======================================================
    // PFlow based Isolation Variables
    //=======================================================

    struct PflowIsolationVariables {
      float chargedHadronIso;                  //charged hadron isolation with dxy,dz match to pv
      float chargedHadronWorstVtxIso;          //max charged hadron isolation when dxy/dz matching to given vtx
      float chargedHadronWorstVtxGeomVetoIso;  //as chargedHadronWorstVtxIso but an additional geometry based veto cone
      float chargedHadronPFPVIso;  //only considers particles assigned to the primary vertex (PV) by particle flow, corresponds to <10_6 chargedHadronIso
      float neutralHadronIso;
      float photonIso;
      float sumEcalClusterEt;  //sum pt of ecal clusters, vetoing clusters part of photon
      float sumHcalClusterEt;  //sum pt of hcal clusters, vetoing clusters part of photon
      PflowIsolationVariables()
          :

            chargedHadronIso(0.),
            chargedHadronWorstVtxIso(0.),
            chargedHadronWorstVtxGeomVetoIso(0.),
            chargedHadronPFPVIso(0.),
            neutralHadronIso(0.),
            photonIso(0.),
            sumEcalClusterEt(0.),
            sumHcalClusterEt(0.) {}
    };

    /// Accessors for Particle Flow Isolation variables
    float chargedHadronIso() const { return pfIsolation_.chargedHadronIso; }
    float chargedHadronWorstVtxIso() const { return pfIsolation_.chargedHadronWorstVtxIso; }
    float chargedHadronWorstVtxGeomVetoIso() const { return pfIsolation_.chargedHadronWorstVtxGeomVetoIso; }
    float chargedHadronPFPVIso() const { return pfIsolation_.chargedHadronPFPVIso; }
    float neutralHadronIso() const { return pfIsolation_.neutralHadronIso; }
    float photonIso() const { return pfIsolation_.photonIso; }

    //backwards compat functions for pat::Photon
    float ecalPFClusterIso() const { return pfIsolation_.sumEcalClusterEt; };
    float hcalPFClusterIso() const { return pfIsolation_.sumHcalClusterEt; };

    /// Get Particle Flow Isolation variables block
    const PflowIsolationVariables& getPflowIsolationVariables() const { return pfIsolation_; }

    /// Set Particle Flow Isolation variables
    void setPflowIsolationVariables(const PflowIsolationVariables& pfisol) { pfIsolation_ = pfisol; }

    struct PflowIDVariables {
      int nClusterOutsideMustache;
      float etOutsideMustache;
      float mva;

      PflowIDVariables()
          :

            nClusterOutsideMustache(-1),
            etOutsideMustache(-999999999.),
            mva(-999999999.)

      {}
    };

    // getters
    int nClusterOutsideMustache() const { return pfID_.nClusterOutsideMustache; }
    float etOutsideMustache() const { return pfID_.etOutsideMustache; }
    float pfMVA() const { return pfID_.mva; }
    // setters
    void setPflowIDVariables(const PflowIDVariables& pfid) { pfID_ = pfid; }

    // go back to run2-like 2 effective depths if desired - depth 1 is the normal depth 1, depth 2 is the sum over the rest
    void hcalToRun2EffDepth();

  private:
    /// check overlap with another candidate
    bool overlap(const Candidate&) const override;
    /// position of seed BasicCluster for shower depth of unconverted photon
    math::XYZPointF caloPosition_;
    /// reference to the PhotonCore
    reco::PhotonCoreRef photonCore_;
    //
    bool pixelSeed_;
    //
    FiducialFlags fiducialFlagBlock_;
    IsolationVariables isolationR04_;
    IsolationVariables isolationR03_;
    ShowerShape showerShapeBlock_;
    ShowerShape full5x5_showerShapeBlock_;
    SaturationInfo saturationInfo_;
    EnergyCorrections eCorrections_;
    MIPVariables mipVariableBlock_;
    PflowIsolationVariables pfIsolation_;
    PflowIDVariables pfID_;
  };

}  // namespace reco

#endif
