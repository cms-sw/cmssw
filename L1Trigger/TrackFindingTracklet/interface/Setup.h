#ifndef L1Trigger_TrackFindingTracklet_Setup_h
#define L1Trigger_TrackFindingTracklet_Setup_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"

#include <vector>
#include <set>
#include <unordered_map>

namespace trklet {

  /*! \class  trklet::Setup
   *  \brief  Class to process and provide run-time constants used by Track Trigger emulators
   *  \author Thomas Schuh 
   *  \date   2020, Apr
   */
  class Setup {
  public:
    // Configuration
    struct Config {
      //  enables emulation of truncation
      bool enableTruncation;
      // use either 4 or 5 parameter fit in simulation
      int simNPar;
      // barrel layer limit r value to partition into PS and 2S region
      double otLimitPSBarrel;
      // barrel layer limit r value to partition into tilted and untilted region
      std::vector<double> otLimitsTiltedR;
      // barrel layer limit |z| value to partition into tilted and untilted region
      std::vector<double> otLimitsTiltedZ;
      // endcap disk limit |z| value to partition into PS and 2S region
      std::vector<double> otLimitsPSDiksZ;
      // endcap disk limit r value to partition into PS and 2S region
      std::vector<double> otLimitsPSDiskR;
      // vector of DTC id indexed by connected IR module id
      std::vector<int> irChannelsIn;
      // f/w frequency in MHz
      double tbFreq;
      // smallest disk stub z position after TrackBuilder in cm
      double tbMinZ;
      // biggest disk stub r position after TrackBuilder in cm
      double tbMaxR;
      // smallest stub radius after TrackBuilder in cm
      double tbInnerRadius;
      // number of seed Types
      int tbNumSeedTypes;
      // number of layers used to form a seed
      int tbNumSeedingLayers;
      // number of layers
      int tbNumLayers;
      // seed types used in tracklet algorithm (position gives int value)
      std::vector<std::string> tbSeedTypes;
      // seeding layers of seed types using default layer id [barrel: 1-6, discs: 11-15]
      std::vector<std::vector<int>> tbSeedTypesSeedLayers;
      // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
      std::vector<std::vector<int>> tbSeedTypesProjectionLayers;
      // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) at TB output
      std::vector<int> tbWidthsR;
      // number of Bits used to represent stubId
      int tbWidthStubId;
      // number of Bits used to represent inv2R
      int tbWidthInv2R;
      // number of Bits used to represent phi0
      int tbWidthPhi0;
      // number of Bits used to represent z0
      int tbWidthZ0;
      // number of Bits used to represent cot
      int tbWidthCot;
      // seed priority during merge
      std::vector<std::string> tmMuxOrder;
      // recalculates track parameter and stub residuals from DTC stubs
      bool drUseDTCStubs;
      // recalculates track parameter and stub residuals from TT stubs
      bool drUseTTStubs;
      // number of comparison modules used in each DR node
      int drNumComparisonModules;
      // min number of shared stubs to identify duplicates
      int drMinIdenticalStubs;
      // number of bits used for stub r - ChosenRofPhi
      int drWidthR;
      // number of bits used for stub phi w.r.t. phi region centre
      int drWidthPhi;
      // number of bits used for stub z
      int drWidthZ;
      // number of Bits used to represent stub phi uncertainty in rad
      int drWidthDPhi;
      // number of Bits used to represent stub z uncertainty in cm
      int drWidthDZ;
      // precision difference in powers of 2 between dPhi and phi
      int drBaseShiftDPhi;
      // precision difference in powers of 2 between dPhi and phi
      int drBaseShiftDZ;
      // simulate KF instead of emulate
      bool kfUseSimulation;
      // max number of tracks a kf worker can process
      int kfMaxTracks;
      // number of layers a fitted track may cross
      int kfNumLayers;
      // required number of layers to form a track
      int kfMinLayers;
      // precision difference in powers of 2 between phi residual and phi position
      int kfBaseShiftPhi;
      // precision difference in powers of 2 between z residual and z position
      int kfBaseShiftZ;
      // number of output channel
      int tqNumChannel;
      // Number of bits used to represent chi2rphi
      int tqWidthChi21;
      // Number of bits used to represent chi2rz
      int tqWidthChi20;
      // Base of chi2rphi gets shifted by that power of 2 w.r.t 1
      int tqBaseShiftChi21;
      // Base of chi2rz gets shifted by that power of 2 w.r.t 1
      int tqBaseShiftChi20;
      // Number of bits used for looked up inverse phi uncertainty squared
      int tqWidthInvV0;
      // Number of bits used for looked up inverse z uncertainty squared
      int tqWidthInvV1;
      // number of bits used for mva
      int tqWidthMVA;
      // scale z0 by this power of 2 for BDT
      int tqScalePowZ0;
      // scale cot by this power of 2 for BDT
      int tqScalePowCot;
      // f/w bin edge integer values to bin mva
      std::vector<int> tqBinEdges;
      // number of bits used for phi0
      int tfpWidthPhi0;
      // umber of bits used for qOverPt
      int tfpWidthInvR;
      // number of bits used for cot(theta)
      int tfpWidthCot;
      // number of bits used for z0
      int tfpWidthZ0;
      // number of output links
      int tfpNumChannel;
    };
    Setup() {}
    Setup(const Config&, const trackerDTC::Setup*);
    ~Setup() = default;

    // TrackerGeometry
    const TrackerGeometry* trackerGeometry() const { return dtc_->trackerGeometry(); }
    // TrackerTopology
    const TrackerTopology* trackerTopology() const { return dtc_->trackerTopology(); }
    // returns global TTStub position
    GlobalPoint stubPosTT(const TTStubRef& ttStubRef) const { return dtc_->stubPosTT(ttStubRef); }
    // returns bit accurate position of a stub from a given tfp region [0-8]
    GlobalPoint stubPosDTC(const tt::FrameStub& frame, int region) const { return dtc_->stubPosDTC(frame, region); }
    // returns bit accurate radial position and phi, z residuals of a TB stub for given TTTrack
    GlobalPoint stubPosTB(const tt::FrameStub&, double) const;
    // returns bit accurate radial position and phi, z residuals of a TB fake seed stub for given TTTrack
    GlobalPoint stubPosTB(const TTStubRef&, double, double) const;
    // sensor module for ttStubRef
    const trackerDTC::SensorModule* sensorModule(const TTStubRef& ttStubRef) const {
      return dtc_->sensorModule(ttStubRef);
    }
    // sensor modules connected to given dtc id
    const std::vector<const trackerDTC::SensorModule*>& dtcModules(int dtcId) const { return dtc_->dtcModules(dtcId); }
    // collection of outer tracker sensor modules
    const std::vector<trackerDTC::SensorModule>& sensorModules() const { return dtc_->sensorModules(); }
    // number of tilted layer rings for given layer
    int numTiltedLayerRing(int layer) const { return dtc_->numTiltedLayerRing(layer); }

    // strip length of outer tracker sensors in cm
    double cbcLength() const { return dtc_->cbcLength(); }
    // strip pitch of outer tracker sensors in cm
    double cbcPitch() const { return dtc_->cbcPitch(); }
    // pixel pitch of outer tracker sensors in cm
    double mpaPitch() const { return dtc_->mpaPitch(); }
    // pixel length of outer tracker sensors in cm
    double mpaLength() const { return dtc_->mpaLength(); }
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double smTiltUncertaintyR() const { return dtc_->smTiltUncertaintyR(); }
    // additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
    double smScattering() const { return dtc_->smScattering(); }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxSlope() const { return dtc_->smTiltApproxSlope(); }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxIntercept() const { return dtc_->smTiltApproxIntercept(); }

    // critical radius in cm defining r-phi Region shape
    double regChosenRofPhi() const { return dtc_->regChosenRofPhi(); }
    // critical radius in cm defining r-z Region shape
    double regChosenRofZ() const { return dtc_->regChosenRofZ(); }
    // number of DTC boards used to readout a detector region
    int regNumDTC() const { return dtc_->regNumDTC(); }
    // half lumi region size in cm defining r-z Region shape
    double regBeamWindowZ() const { return dtc_->regBeamWindowZ(); }
    // defining r-z Region shape in cm
    double regMaxEta() const { return dtc_->regMaxEta(); }
    // min pt in GeV defining r-phi Region shape
    double regMinPt() const { return dtc_->regMinPt(); }
    // single region phiT range in rad [2pi/9]
    double regRangePhiT() const { return dtc_->regRangePhiT(); }

    // number of phi slices the outer tracker readout is organized in
    int sysNumRegion() const { return dtc_->sysNumRegion(); }
    // number of detector layer a particle may cross
    int sysNumLayer() const { return dtc_->sysNumLayer(); }
    // in T
    double sysBField() const { return dtc_->sysBField(); }
    // half length of outer tracker in cm
    double sysHalfLength() const { return dtc_->sysHalfLength(); }
    // converts GeV in 1/cm
    double sysInvPtToDphi() const { return dtc_->sysInvPtToDphi(); }
    // smallest stub radius after TrackBuilder in cm
    double sysInnerRadius() const { return dtc_->sysInnerRadius(); }
    // outer radius of outer tracker in cm
    double sysOuterRadius() const { return dtc_->sysOuterRadius(); }
    // number of frames which can be send out of DTC/TFP per event
    int sysNumFrames() const { return dtc_->sysNumFrames(); }
    // number of Slots in used ATCA crates
    int sysNumATCASlot() const { return dtc_->sysNumATCASlot(); }

    // number of bits used for stub r - ChosenRofPhi
    int glWidthR() const { return dtc_->glWidthR(); }
    // number of bits used for stub phi w.r.t. phi region centre
    int glWidthPhi() const { return dtc_->glWidthPhi(); }
    // number of bits used for stub z
    int glWidthZ() const { return dtc_->glWidthZ(); }

    // mean z of outer tracker endcap disks
    double stubDiskZ(int disk) const { return dtc_->stubDiskZ(disk); }
    // mean r of outer tracker endcap disk
    double stubDiskR(int disk) const { return dtc_->stubDiskR(disk, 0); }
    // range of output stub phi in rad
    double stubRangePhi() const { return dtc_->stubRangePhi(); }
    // center radius of outer tracker endcap 2S diks strips
    double stubDiskR(int layerId, int r) const { return dtc_->stubDiskR(layerId, r); }
    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double stubBaseR(trackerDTC::SensorModule::Type type) const { return dtc_->stubBaseR(type); }
    // mean radius of outer tracker barrel layer
    double stubLayerR(int layer) const { return dtc_->stubLayerR(layer); }

    // smallest address width of an BRAM18 configured as broadest simple dual port memory
    int widthAddrBRAM18() const { return dtc_->widthAddrBRAM18(); }
    // width of the 'A' port of an DSP slice using biased binary
    int widthDSPau() const { return dtc_->widthDSPau(); }
    // width of the 'A' port of an DSP slice using biased two's complement
    int widthDSPab() const { return dtc_->widthDSPab(); }
    // width of the 'B' port of an DSP slice using biased binary
    int widthDSPbu() const { return dtc_->widthDSPbu(); }
    // width of the 'B' port of an DSP slice using biased two's complement
    int widthDSPbb() const { return dtc_->widthDSPbb(); }
    // enables simulation of truncation
    bool enableTruncation() const { return config_.enableTruncation; }
    // number of frames which can be processed internally using high clock frequency
    int numFrames() const { return numFrames_; }

    // use either 4 or 5 parameter fit in simulation
    int simNPar() const { return config_.simNPar; }

    // number of seed Types
    int tbNumSeedTypes() const { return config_.tbNumSeedTypes; }
    // smallest disk stub z position after TrackBuilder in cm
    double tbMinZ() const { return config_.tbMinZ; }
    // smallest stub radius after TrackBuilder in cm
    double tbInnerRadius() const { return config_.tbInnerRadius; }
    // largest possible cotTheta
    double tbMaxCot() const { return tbMaxCot_; }
    // number of layers used to form a seed
    int tbNumSeedingLayers() const { return config_.tbNumSeedingLayers; }
    // number of layers
    int tbNumLayers() const { return config_.tbNumLayers; }
    // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
    int tbNumProjectionLayers(int seedType) const { return config_.tbSeedTypesProjectionLayers[seedType].size(); }
    // layers a seed types can project to using default layer id [barrel: 1-6, discs: 11-15]
    const std::vector<int>& tbProjectionLayers(int seedType) const {
      return config_.tbSeedTypesProjectionLayers[seedType];
    }
    // seeding layers of seed types using default layer id [barrel: 1-6, discs: 11-15]
    const std::vector<int>& tbSeedLayers(int seedType) const { return config_.tbSeedTypesSeedLayers[seedType]; }
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) after TrackBuilder
    int tbWidthR(trackerDTC::SensorModule::Type type) const { return config_.tbWidthsR.at(type); }
    // number of Bits used to represent stubId
    int tbWidthStubId() const { return config_.tbWidthStubId; }
    // number of Bits used to represent inv2R
    int tbWidthInv2R() const { return config_.tbWidthInv2R; }
    // number of Bits used to represent phi0
    int tbWidthPhi0() const { return config_.tbWidthPhi0; }
    // number of Bits used to represent z0
    int tbWidthZ0() const { return config_.tbWidthZ0; }
    // number of Bits used to represent cot
    int tbWidthCot() const { return config_.tbWidthCot; }
    double tbBaseInv2R() const { return tbBaseInv2R_; }
    double tbBasePhi0() const { return tbBasePhi0_; }
    double tbBaseCot() const { return tbBaseCot_; }
    double tbBaseZ0() const { return tbBaseZ0_; }
    double tbBaseR() const { return tbBaseR_; }
    double tbBasePhi() const { return tbBasePhi_; }
    double tbBasePhi(int layer) const { return tbBasePhis_[layer]; }
    double tbBaseZ() const { return tbBaseZ_; }
    double tbBaseZ(int layer) const { return tbBaseZs_[layer]; }
    int tbWidthZ() const { return tbWidthZ_; }
    int tbWidthR() const { return tbWidthR_; }
    int tbWidthPhi() const { return tbWidthPhi_; }
    // number of bits used to represent seed type
    int tbWidthSeedType() const { return tbWidthSeedType_; }

    // mux order of seed types
    const std::vector<int>& tmMuxOrder() const { return tmMuxOrder_; }
    // number of layers per track
    int tmNumLayers() const { return tmNumLayers_; }
    // number of bits used to represent trackelt residual object
    int tmWidthResid() const { return tmWidthResid_; }

    // number of comparison modules used in each DR node
    int drNumComparisonModules() const { return config_.drNumComparisonModules; }
    // min number of shared stubs to identify duplicates
    int drMinIdenticalStubs() const { return config_.drMinIdenticalStubs; }
    // number of bits used for stub r - ChosenRofPhi
    int drWidthR() const { return config_.drWidthR; }
    // number of bits used for stub phi w.r.t. phi region centre
    int drWidthPhi() const { return config_.drWidthPhi; }
    // number of bits used for stub z
    int drWidthZ() const { return config_.drWidthZ; }
    // number of Bits used to represent stub phi uncertainty in rad
    int drWidthDPhi() const { return config_.drWidthDPhi; }
    // number of Bits used to represent stub z uncertainty in cm
    int drWidthDZ() const { return config_.drWidthDZ; }
    // precision difference in powers of 2 between dPhi and phi
    int drBaseShiftDPhi() const { return config_.drBaseShiftDPhi; }
    // precision difference in powers of 2 between dPhi and phi
    int drBaseShiftDZ() const { return config_.drBaseShiftDZ; }
    //recalculates track parameter and stub residuals from DTC stubs
    bool drUseDTCStubs() const { return config_.drUseDTCStubs; }
    //recalculates track parameter and stub residuals from DTC stubs
    bool drUseTTStubs() const { return config_.drUseTTStubs; }
    // number of layers per track
    int drNumLayers() const { return drNumLayers_; }
    // precision of internal inversere cot(theta)
    double drBaseInvCot() const { return drBaseInvCot_; }

    // simulate KF instead of emulate
    bool kfUseSimulation() const { return config_.kfUseSimulation; }
    // max number of tracks a kf worker can process
    int kfMaxTracks() const { return config_.kfMaxTracks; }
    // number of layers a fitted track may cross
    int kfNumLayers() const { return config_.kfNumLayers; }
    // required number of layers to form a track
    int kfMinLayers() const { return config_.kfMinLayers; }
    // precision difference in powers of 2 between phi residual and phi position
    int kfBaseShiftPhi() const { return config_.kfBaseShiftPhi; }
    // precision difference in powers of 2 between z residual and z position
    int kfBaseShiftZ() const { return config_.kfBaseShiftZ; }
    // number of projection layers per track
    int kfNumProj() const { return kfNumProj_; }
    // minimum number of projection layers to form a track
    int kfMinProj() const { return kfMinProj_; }

    // number of output channel
    int tqNumChannel() const { return config_.tqNumChannel; }
    // Number of bits used to represent chi2rphi
    int tqBaseShiftChi20() const { return config_.tqBaseShiftChi20; }
    // Number of bits used to represent chi2rz
    int tqBaseShiftChi21() const { return config_.tqBaseShiftChi21; }
    // Base of chi2rphi gets shifted by that power of 2 w.r.t 1
    int tqWidthChi20() const { return config_.tqWidthChi20; }
    // Base of chi2rz gets shifted by that power of 2 w.r.t 1
    int tqWidthChi21() const { return config_.tqWidthChi21; }
    // Number of bits used for looked up inverse phi uncertainty squared
    int tqWidthMVA() const { return config_.tqWidthMVA; }
    // Number of bits used for looked up inverse z uncertainty squared
    int tqWidthInvV0() const { return config_.tqWidthInvV0; }
    // number of bits used for mva
    int tqWidthInvV1() const { return config_.tqWidthInvV1; }
    // scale z0 by this factor for BDT
    double tqScaleFactorZ0() const { return tqScaleFactorZ0_; }
    // scale cot by this factor for BDT
    double tqScaleFactorCot() const { return tqScaleFactorCot_; }
    // f/w bin edge integer values to bin mva
    const std::vector<int>& tqBinEdges() const { return config_.tqBinEdges; }

    // number of bist used for phi0
    int tfpWidthPhi0() const { return config_.tfpWidthPhi0; }
    // umber of bist used for invR
    int tfpWidthInvR() const { return config_.tfpWidthInvR; }
    // number of bist used for cot(theta)
    int tfpWidthCot() const { return config_.tfpWidthCot; }
    // number of bist used for z0
    int tfpWidthZ0() const { return config_.tfpWidthZ0; }
    // number of output links
    int tfpNumChannel() const { return config_.tfpNumChannel; }

  private:
    // DTC setup
    const trackerDTC::Setup* dtc_;
    // configuration
    Config config_;
    // number of frames betwen 2 resets of 18 BX packets
    int numFrames_;
    // largest possible cotTheta
    double tbMaxCot_;
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) after TrackBuilder
    std::vector<int> tbWidthsR_;
    double tbBaseInv2R_;
    double tbBasePhi0_;
    double tbBaseCot_;
    double tbBaseZ0_;
    double tbBaseR_;
    double tbBasePhi_;
    double tbBaseZ_;
    std::vector<double> tbBasePhis_;
    std::vector<double> tbBaseZs_;
    int tbWidthZ_;
    int tbWidthR_;
    int tbWidthPhi_;
    // number of bits used to represent seed type
    int tbWidthSeedType_;
    // number of bits used to represent trackelt residual object
    int tmWidthResid_;
    double tmBaseCot_;
    // mux order of seed types
    std::vector<int> tmMuxOrder_;
    // number of layers per track
    int tmNumLayers_;
    // precision of internal inversere cot(theta)
    double drBaseInvCot_;
    // number of layers per track
    int drNumLayers_;
    // number of projection layers per track
    int kfNumProj_;
    // minimum number of projection layers to form a track
    int kfMinProj_;
    // scale z0 by this factor for BDT
    double tqScaleFactorZ0_;
    // scale cot by this factor for BDT
    double tqScaleFactorCot_;
  };

}  // namespace trklet

EVENTSETUP_DATA_DEFAULT_RECORD(trklet::Setup, trackerDTC::SetupRcd);

#endif
