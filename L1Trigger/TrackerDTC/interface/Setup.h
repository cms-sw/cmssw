#ifndef L1Trigger_TrackerDTC_Setup_h
#define L1Trigger_TrackerDTC_Setup_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/SetupRcd.h"

#include <vector>
#include <string>

namespace trackerDTC {

  typedef TTStubAlgorithm<Ref_Phase2TrackerDigi_> StubAlgorithm;
  typedef TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> StubAlgorithmOfficial;

  /*! \class  trackerDTC::Setup
   *  \brief  Class to provide constants, data formats and algorithms used by DTC emulator
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class Setup {
  public:
    // TTStubAlgorithm configuration
    struct ConfigTTStubAlgorithm {
      std::vector<double> nTiltedRings;
      std::vector<double> barrelCut;
      std::vector<std::vector<double>> tiltedBarrelCutSet;
      std::vector<std::vector<double>> endcapCutSet;
    };
    // Configuration
    struct Config {
      // enables printing of board constants
      bool printConstants;
      // enables printing of bend encoding
      bool printEncodingBend;
      // dtcs to be printed [0 - 215]; empty means all
      std::vector<int> printIDs;
      // path of prints
      std::string printPath;
      // number of rows read out
      int cbcNumRow;
      // number of coloumns read out
      int cbcNumCol;
      // number of stubs collected
      int cbcNumStub;
      // number of events used to collect stubs
      int cbcNumBX;
      // number of bits used for internal stub bend
      int cbcWidthBend;
      //strip pitch of outer tracker sensors in cm
      double cbcPitch;
      // strip length of outer tracker sensors in cm
      double cbcLength;
      // number of rows read out
      int mpaNumRow;
      // number of coloumns read out
      int mpaNumCol;
      // number of stubs collected
      int mpaNumStub;
      // number of events used to collect stubs
      int mpaNumBX;
      // number of bits used for internal stub bend
      int mpaWidthBend;
      // pixel pitch of outer tracker sensors in cm
      double mpaPitch;
      // pixel length of outer tracker sensors in cm
      double mpaLength;
      // number of events collected
      int cicNumBX;
      // number of stubs collected for 5 gbps config
      int cicNumStub5g;
      // number of stubs collected for 10 gbps config
      int cicNumStub10g;
      // number of MPAs/CBCs read out
      int cicNumFEC;
      // number of CICs per sensor module
      int smNumCIC;
      // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
      double smTiltApproxSlope;
      // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
      double smTiltApproxIntercept;
      // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
      double smTiltUncertaintyR;
      // additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
      double smScattering;
      // bend uncertainty in pitch units defining stub pt uncertainty
      double smBendCut;
      // # average ClusterWidths for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
      std::vector<double> smClusterWidth;
      // additional phi uncertainties in rad for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
      std::vector<double> smAddPhiUncertainty;
      // max number of layer connected to one DTC
      int dtcNumLayer;
      // max number of sensor modules connected to one DTC board
      int dtcNumModule;
      // DTC clock frequency in MHz
      double dtcFreq;
      // number of DTC boards used to readout a detector region
      int regNumDTC;
      // number of TFP boards used to process a processing region
      int regNumTFP;
      // min pt in GeV defining r-phi Region shape
      double regMinPt;
      // in cm defining r-phi Region shape
      double regMaxD0;
      // half lumi region size in cm defining r-z Region shape
      double regBeamWindowZ;
      // defining r-z Region shape in cm
      double regMaxEta;
      // critical radius in cm defining r-phi Region shape
      double regChosenRofPhi;
      // critical radius in cm defining r-z Region shape
      double regChosenRofZ;
      // total number of modules
      int sysNumModule;
      // number of phi slices the outer tracker readout is organized in
      int sysNumRegion;
      // number of regions a reconstructable particles may cross
      int sysNumOverlap;
      // number of Slots in used ATCA crates
      int sysNumATCASlot;
      // slot number changing from PS to 2S
      int sysSlotLimitPS;
      // slot number changing from 10 gbps to 5gbps
      int sysSlotLimit10gbps;
      // number of barrel layer
      int sysNumBarrelLayer;
      // number of barrel ps layer
      int sysNumBarrelLayerPS;
      // needed gap between events of emp-infrastructure firmware
      int sysNumFramesInfra;
      // number of detector layer a particle may cross
      int sysNumLayer;
      // in e8 m/s
      double sysSpeedOfLight;
      // in T
      double sysBField;
      // outer radius of outer tracker in cm
      double sysOuterRadius;
      // inner radius of outer tracker in cm
      double sysInnerRadius;
      // half length of outer tracker in cm
      double sysHalfLength;
      // LHC bunch crossing rate in MHz
      double sysLhcFreq;
      // precision of internal stub bend in pitch units
      double feBaseBend;
      // precision of internal stub column in pitch units
      double feBaseCol;
      // precision of internal stub row in pitch units
      double feBaseRow;
      // number of bits used for stub r - ChosenRofPhi
      int glWidthR;
      // number of bits used for stub phi w.r.t. phi region centre
      int glWidthPhi;
      // number of bits used for stub z
      int glWidthZ;
      // number of outer PS rings for disk 1, 2, 3, 4, 5
      std::vector<int> stubNumRingsPS;
      // mean radius of outer tracker barrel layer
      std::vector<double> stubLayerRs;
      // mean z of outer tracker endcap disks
      std::vector<double> stubDiskZs;
      // center radius of outer tracker endcap 2S diks strips
      std::vector<std::vector<double>> stubDisk2SRs;
      // number of bits used for stub negDisk boolean, determined by if in neg. or pos. z region (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsND;
      // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsR;
      // number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsZ;
      // number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsPhi;
      // number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsAlpha;
      // number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<int> stubWidthsBend;
      // range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<double> stubRangesR;
      // range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<double> stubRangesZ;
      // range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
      std::vector<double> stubRangesAlpha;
      // radial offset in cm applied to disk PS stubs
      double stubOffsetRDiskPS;
      // in GeV, used to define output stub data format
      double stubMinPt;
      // fifo addr width in stub router firmware
      int unWidthAddr;
      // number of parallel worker
      int unNumNode;
      // repacking factor denominator
      int reIn;
      // repacking factor numerator
      int reOut;
      // enables simulation of truncation
      bool fwEnableTruncation;
      // width of the 'A' port of an DSP slice
      int fwWidthDSPa;
      // width of the 'B' port of an DSP slice
      int fwWidthDSPb;
      // width of the 'C' port of an DSP slice
      int fwWidthDSPc;
      // smallest address width of an BRAM36 configured as broadest simple dual port memory
      int fwWidthAddrBRAM36;
      // smallest address width of an BRAM18 configured as broadest simple dual port memory
      int fwWidthAddrBRAM18;
    };
    Setup() {}
    Setup(const Config&,
          const TrackerGeometry&,
          const TrackerTopology&,
          const TrackerDetToDTCELinkCablingMap&,
          const StubAlgorithmOfficial&,
          const ConfigTTStubAlgorithm&);
    ~Setup() = default;
    // returns global TTStub position
    GlobalPoint stubPosTT(const TTStubRef&) const;
    // returns bit accurate position of a stub from a given tfp region [0-8]
    GlobalPoint stubPosDTC(const tt::FrameStub&, int) const;
    // sensor modules connected to given dtc id
    const std::vector<const SensorModule*>& dtcModules(int dtcId) const { return dtcModules_[dtcId]; }
    // sensor module for ttStubRef
    const SensorModule* sensorModule(const TTStubRef& ttStubRef) const {
      return sensorModule(ttStubRef->getDetId() + 1);
    }
    // sensor module for det id
    const SensorModule* sensorModule(const DetId& detId) const { return detIdToSensorModule_.find(detId)->second; }
    // collection of outer tracker sensor modules
    const std::vector<SensorModule>& sensorModules() const { return sensorModules_; }
    // TrackerGeometry
    const TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    // TrackerTopology
    const TrackerTopology* trackerTopology() const { return trackerTopology_; }
    // number of tilted layer rings for given layer
    int numTiltedLayerRing(int layer) const { return nTiltedRings_[layer]; }
    // number of rows read out
    int cbcNumRow() const { return config_.cbcNumRow; }
    // number of coloumns read out
    int cbcNumCol() const { return config_.cbcNumCol; }
    // number of stubs collected
    int cbcNumStub() const { return config_.cbcNumStub; }
    //strip pitch of outer tracker sensors in cm
    double cbcPitch() const { return config_.cbcPitch; }
    // strip length of outer tracker sensors in cm
    double cbcLength() const { return config_.cbcLength; }
    // number of rows read out
    int mpaNumRow() const { return config_.mpaNumRow; }
    // number of coloumns read out
    int mpaNumCol() const { return config_.mpaNumCol; }
    // number of stubs collected
    int mpaNumStub() const { return config_.mpaNumStub; }
    // number of events used to collect stubs
    int mpaNumBX() const { return config_.mpaNumBX; }
    // pixel pitch of outer tracker sensors in cm
    double mpaPitch() const { return config_.mpaPitch; }
    // pixel length of outer tracker sensors in cm
    double mpaLength() const { return config_.mpaLength; }
    // number of events collected
    int cicNumBX() const { return config_.cicNumBX; }
    // number of stubs collected for 5 gbps config
    int cicNumStub5g() const { return config_.cicNumStub5g; }
    // number of stubs collected for 10 gbps config
    int cicNumStub10g() const { return config_.cicNumStub10g; }
    // number of MPAs/CBCs read out
    int cicNumFEC() const { return config_.cicNumFEC; }
    // number of CICs per sensor module
    int smNumCIC() const { return config_.smNumCIC; }
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double smTiltUncertaintyR() const { return config_.smTiltUncertaintyR; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxSlope() const { return config_.smTiltApproxSlope; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxIntercept() const { return config_.smTiltApproxIntercept; }
    // additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
    double smScattering() const { return config_.smScattering; }
    // bend uncertainty in pitch units defining stub pt uncertainty
    double smBendCut() const { return config_.smBendCut; }
    // average ClusterWidths for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
    double smClusterWidth(int type) const { return config_.smClusterWidth[type]; }
    // additional phi uncertainties in rad for ("Barrel2S", "BarrelPSFlat", "BarrelPSTilted", "Disk2S", "DiskPS" )
    double smAddPhiUncertainty(int type) const { return config_.smAddPhiUncertainty[type]; }
    // max number of sensor modules connected to one DTC board
    int dtcNumModule() const { return config_.dtcNumModule; }
    // DTC clock frequency in MHz
    double dtcFreq() const { return config_.dtcFreq; }
    // number of TFPs connected to one DTC
    int dtcNumTFP() const { return dtcNumTFP_; }
    // number of DTC boards used to readout a detector region
    int regNumDTC() const { return config_.regNumDTC; }
    // number of TFPs processing one region
    int regNumTFP() const { return config_.regNumTFP; }
    // min pt in GeV defining r-phi Region shape
    double regMinPt() const { return config_.regMinPt; }
    // critical radius in cm defining r-phi Region shape
    double regChosenRofPhi() const { return config_.regChosenRofPhi; }
    // critical radius in cm defining r-z Region shape
    double regChosenRofZ() const { return config_.regChosenRofZ; }
    // half lumi region size in cm defining r-z Region shape
    double regBeamWindowZ() const { return config_.regBeamWindowZ; }
    // defining r-z Region shape in cm
    double regMaxEta() const { return config_.regMaxEta; }
    // pt cut
    double regMaxInv2R() const { return regMaxInv2R_; }
    // "eta" cut
    double regMaxZT() const { return regMaxZT_; }
    // single region phiT range in rad [2pi/9]
    double regRangePhiT() const { return regRangePhiT_; }
    // in T
    double sysBField() const { return config_.sysBField; }
    // outer radius of outer tracker in cm
    double sysOuterRadius() const { return config_.sysOuterRadius; }
    // inner radius of outer tracker in cm
    double sysInnerRadius() const { return config_.sysInnerRadius; }
    // half length of outer tracker in cm
    double sysHalfLength() const { return config_.sysHalfLength; }
    // number of phi slices the outer tracker readout is organized in
    int sysNumRegion() const { return config_.sysNumRegion; }
    // number of regions a reconstructable particles may cross
    int sysNumOverlap() const { return config_.sysNumOverlap; }
    // number of Slots in used ATCA crates
    int sysNumATCASlot() const { return config_.sysNumATCASlot; }
    // slot number changing from PS to 2S
    int sysSlotLimitPS() const { return config_.sysSlotLimitPS; }
    // number of detector layer a particle may cross
    int sysNumLayer() const { return config_.sysNumLayer; }
    // total number of dtcs
    int sysNumDTC() const { return sysNumDTC_; }
    // needed gap between events of emp-infrastructure firmware
    int sysNumFramesInfra() const { return config_.sysNumFramesInfra; }
    // LHC bunch crossing rate in MHz
    double sysLhcFreq() const { return config_.sysLhcFreq; }
    // converts GeV in 1/cm
    double sysInvPtToDphi() const { return sysInvPtToDphi_; }
    // number of frames which can be send out of DTC per event
    int sysNumFrames() const { return sysNumFrames_; }
    // number of bits used for internal stub bend
    int feWidthBend() const { return feWidthBend_; }
    // number of bits used for internal stub column from CIC
    int feWidthCol() const { return feWidthCol_; }
    // number of bits used for internal stub row from FEC
    int feWidthRow() const { return feWidthRow_; }
    // number of bits used for internal FEC identifier
    int feWidthFEC() const { return feWidthFEC_; }
    // number of bits used for internal stub bx
    int feWidthBX() const { return feWidthBX_; }
    // number of bits used the represent a FE stub data
    int fePosValid() const { return fePosValid_; }
    // precision of internal stub bend in pitch units
    double feBaseBend() const { return config_.feBaseBend; }
    // precision of internal stub column in pitch units
    double feBaseCol() const { return config_.feBaseCol; }
    // precision of internal stub row in pitch units
    double feBaseRow() const { return config_.feBaseRow; }
    // number of valid frames in 8BX boxcar
    int feNumFrames() const { return feNumFrames_; }
    // number of bits used for stub r - ChosenRofPhi
    int glWidthR() const { return config_.glWidthR; }
    // number of bits used for stub phi w.r.t. phi region centre
    int glWidthPhi() const { return config_.glWidthPhi; }
    // number of bits used for stub z
    int glWidthZ() const { return config_.glWidthZ; }
    // number of bits used for stub inv2R. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int glWidthInv2R() const { return glWidthInv2R_; }
    // global stub r precision in cm
    double glBaseR() const { return glBaseR_; }
    // global stub phi precision in rad
    double glBasePhi() const { return glBasePhi_; }
    // global stub z precision in cm
    double glBaseZ() const { return glBaseZ_; }
    // global stub inv2R precision in 1/cm
    double glBaseInv2R() const { return glBaseInv2R_; }
    // global stub partial r / row precision in cm
    double glBaseRM() const { return glBaseRM_; }
    // global stub partial r precision in cm
    double glBaseRC() const { return glBaseRC_; }
    // global stub partial phi / row precision in rad
    double glBasePhiM() const { return glBasePhiM_; }
    // global stub partial phi precision in rad
    double glBasePhiC() const { return glBasePhiC_; }
    // number of outer PS rings for disk 1, 2, 3, 4, 5
    int stubNumRingsPS(int layerId) const { return config_.stubNumRingsPS[layerId]; }
    // mean radius of outer tracker barrel layer
    double stubLayerR(int layer) const { return config_.stubLayerRs[layer]; }
    // mean z of outer tracker endcap disks
    double stubDiskZ(int disk) const { return config_.stubDiskZs[disk]; }
    // mean r of outer tracker endcap disk
    double stubDiskR(int disk, int ring) const { return config_.stubDisk2SRs[disk][ring]; }
    // number of bits used for stub negDisk boolean, determined by if in neg. or pos. z region (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthND(SensorModule::Type type) const { return config_.stubWidthsND[type]; }
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthR(SensorModule::Type type) const { return config_.stubWidthsR[type]; }
    // number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthZ(SensorModule::Type type) const { return config_.stubWidthsZ[type]; }
    // number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthPhi(SensorModule::Type type) const { return config_.stubWidthsPhi[type]; }
    // number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthAlpha(SensorModule::Type type) const { return config_.stubWidthsAlpha[type]; }
    // number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    int stubWidthBend(SensorModule::Type type) const { return config_.stubWidthsBend[type]; }
    // number of bits used to represent encoded layer id [2]
    int stubWidthLayerId() const { return stubWidthLayerId_; }
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    int stubNumUnusedBits(SensorModule::Type type) const { return stubNumsUnusedBits_[type]; }
    // radial offset in cm applied to disk PS stubs
    double stubOffsetRDiskPS() const { return config_.stubOffsetRDiskPS; }
    // range of output stub phi in rad
    double stubRangePhi() const { return stubRangePhi_; }
    // base transformed precision in cm
    double stubBaseR() const { return stubBaseR_; }
    // base transformed precision in rad
    double stubBasePhi() const { return stubBasePhi_; }
    //base transformed precision in cm
    double stubBaseZ() const { return stubBaseZ_; }
    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double stubBaseR(SensorModule::Type type) const { return stubBasesR_[type]; }
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double stubBaseZ(SensorModule::Type type) const { return stubBasesZ_[type]; }
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    double stubBasePhi(SensorModule::Type type) const { return stubBasesPhi_[type]; }
    // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    double stubBaseAlpha(SensorModule::Type type) const { return stubBasesAlpha_[type]; }
    // in GeV, used to define output stub data format
    double stubMinPt() const { return config_.stubMinPt; }
    // fifo depth in stub router firmware
    int unDepth() const { return unDepth_; }
    // number of 8bx worker
    int tmp8NumNodes() const { return tmp8NumNodes_; }
    // number of 8bx worker inputs
    int tmp8NumInputs() const { return tmp8NumInputs_; }
    // number of 8bx worker outputs
    int tmp8NumOutputs() const { return tmp8NumOutputs_; }
    // total number of 8bx outpus
    int tmp8NumChannel() const { return tmp8NumChannel_; }
    // unnmber of frames available in 8 bx
    int tmp8NumFrames() const { return tmp8NumFrames_; }
    // number of 12bx worker
    int tmp12NumNodes() const { return tmp12NumNodes_; }
    // number of 12bx worker inputs
    int tmp12NumInputs() const { return tmp12NumInputs_; }
    // number of 12bx worker outputs
    int tmp12NumOutputs() const { return tmp12NumOutputs_; }
    // total number of 12bx outpus
    int tmp12NumChannel() const { return tmp12NumChannel_; }
    // unnmber of frames available in 12 bx
    int tmp12NumFrames() const { return tmp12NumFrames_; }
    // number of 18bx worker
    int tmp18NumNodes() const { return tmp18NumNodes_; }
    // number of 18bx worker inputs
    int tmp18NumInputs() const { return tmp18NumInputs_; }
    // number of 18bx worker outputs
    int tmp18NumOutputs() const { return tmp18NumOutputs_; }
    // total number of 18bx outpus
    int tmp18NumChannel() const { return tmp18NumChannel_; }
    // number of frames available in 18 bx
    int tmp18NumFrames() const { return tmp18NumFrames_; }
    // enables simulation of truncation
    bool enableTruncation() const { return config_.fwEnableTruncation; }
    // width of the 'A' port of an DSP slice using biased binary
    int widthDSPau() const { return fwWidthDSPau_; }
    // width of the 'A' port of an DSP slice using biased two's complement
    int widthDSPab() const { return fwWidthDSPab_; }
    // width of the 'B' port of an DSP slice using biased binary
    int widthDSPbu() const { return fwWidthDSPbu_; }
    // width of the 'B' port of an DSP slice using biased two's complement
    int widthDSPbb() const { return fwWidthDSPbb_; }
    // smallest address width of an BRAM18 configured as broadest simple dual port memory
    int widthAddrBRAM18() const { return config_.fwWidthAddrBRAM18; }

  private:
    // converts tk layout id into dtc id
    int dtcId(int tklId) const;
    // TrackerGeometry
    const TrackerGeometry* trackerGeometry_;
    // TrackerTopology
    const TrackerTopology* trackerTopology_;
    // number of tilted layer rings per layer
    std::vector<double> nTiltedRings_;
    // Configuration
    Config config_;
    // dtcs to be printed [0 - 215]; empty means all
    std::vector<int> printIDs_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for ps modules
    std::vector<std::vector<double>> encodingsBendPS_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for 2s modules
    std::vector<std::vector<double>> encodingsBend2S_;
    // outer index = module window size, inner index = encoded bend, inner value = degraded bend, for ps modules
    std::vector<std::vector<double>> degradedBendPS_;
    // outer index = module window size, inner index = encoded bend, inner value = degraded bend, for 2s modules
    std::vector<std::vector<double>> degradedBend2S_;
    // number of TFPs connected to one DTC
    int dtcNumTFP_;
    // pt cut
    double regMaxInv2R_;
    // "eta" cut
    double regMaxZT_;
    // single region phiT range in rad [2pi/9]
    double regRangePhiT_;
    // total number of dtcs
    int sysNumDTC_;
    // converts GeV in 1/cm
    double sysInvPtToDphi_;
    // number of frames which can be send out of DTC per event
    int sysNumFrames_;
    //  largest fe stub window
    int feMaxWindowSize_;
    // number of bits used for internal stub bx
    int feWidthBX_;
    // number of bits used for internal stub bend
    int feWidthBend_;
    // number of bits used for internal stub column from CIC
    int feWidthCol_;
    // number of bits used for internal CIC identifier
    int feWidthCIC_;
    // number of bits used for internal stub row from FEC
    int feWidthRow_;
    // number of bits used for internal FEC identifier
    int feWidthFEC_;
    // number of bits used the represent a FE stub data
    int fePosValid_;
    // number of valid frames in 8BX boxcar
    int feNumFrames_;
    // number of bits used for stub inv2R. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int glWidthInv2R_;
    // global stub r precision in cm
    double glBaseR_;
    // global stub phi precision in rad
    double glBasePhi_;
    // global stub z precision in cm
    double glBaseZ_;
    // global stub inv2R precision in 1/cm
    double glBaseInv2R_;
    // global stub partial r / row precision in cm
    double glBaseRM_;
    // global stub partial r precision in cm
    double glBaseRC_;
    // global stub partial phi / row precision in rad
    double glBasePhiM_;
    // global stub partial phi precision in rad
    double glBasePhiC_;
    // range of output stub phi in rad
    double stubRangePhi_;
    // number of bits used to represent encoded layer id [2]
    int stubWidthLayerId_;
    // base transformed precision in cm
    double stubBaseR_;
    // base transformed precision in rad
    double stubBasePhi_;
    //base transformed precision in cm
    double stubBaseZ_;
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> stubNumsUnusedBits_;
    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> stubBasesR_;
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> stubBasesZ_;
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> stubBasesPhi_;
    // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> stubBasesAlpha_;
    // fifo depth in stub router firmware
    int unDepth_;
    // number of 8bx worker
    int tmp8NumNodes_;
    // number of 8bx worker inputs
    int tmp8NumInputs_;
    // number of 8bx worker outputs
    int tmp8NumOutputs_;
    // total number of 8bx outpus
    int tmp8NumChannel_;
    // unnmber of frames available in 8 bx
    int tmp8NumFrames_;
    // number of 12bx worker
    int tmp12NumNodes_;
    // number of 12bx worker inputs
    int tmp12NumInputs_;
    // number of 12bx worker outputs
    int tmp12NumOutputs_;
    // total number of 12bx outpus
    int tmp12NumChannel_;
    // unnmber of frames available in 12 bx
    int tmp12NumFrames_;
    // number of 18bx worker
    int tmp18NumNodes_;
    // number of 18bx worker inputs
    int tmp18NumInputs_;
    // number of 18bx worker outputs
    int tmp18NumOutputs_;
    // total number of 8bx outpus
    int tmp18NumChannel_;
    // unnmber of frames available in 8 bx
    int tmp18NumFrames_;
    // width of the 'A' port of an DSP slice using biased binary
    int fwWidthDSPau_;
    // width of the 'A' port of an DSP slice using biased two's complement
    int fwWidthDSPab_;
    // width of the 'B' port of an DSP slice using biased binary
    int fwWidthDSPbu_;
    // width of the 'B' port of an DSP slice using biased two's complement
    int fwWidthDSPbb_;
    // collection of outer tracker sensor modules
    std::vector<SensorModule> sensorModules_;
    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<const SensorModule*>> dtcModules_;
    // hepler to convert Stubs quickly
    std::unordered_map<DetId, const SensorModule*> detIdToSensorModule_;
  };

}  // namespace trackerDTC

EVENTSETUP_DATA_DEFAULT_RECORD(trackerDTC::Setup, trackerDTC::SetupRcd);

#endif
