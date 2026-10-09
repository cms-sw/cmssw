#ifndef L1Trigger_TrackerTFP_Setup_h
#define L1Trigger_TrackerTFP_Setup_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"

#include <vector>

namespace trackerTFP {

  /*! \class  trackerTFP::Setup
   *  \brief  Class to process and provide run-time constants used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class Setup {
  public:
    // Configuration
    struct Config {
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
      // number of phi sectors used in hough transform
      int gpNumBinsPhiT;
      // number of eta sectors used in hough transform
      int gpNumBinsZT;
      // fifo depth in stub router firmware
      int gpDepthMemory;
      // number of inv2R bins used in hough transform
      int htNumBinsInv2R;
      // number of phiT bins used in hough transform
      int htNumBinsPhiT;
      // required number of stub layers to form a candidate
      int htMinLayers;
      // internal fifo depth
      int htDepthMemory;
      // number of finer inv2R bins inside HT bin
      int ctbNumBinsInv2R;
      // number of finer phiT bins inside HT bin
      int ctbNumBinsPhiT;
      // number of used cot bins inside GP ZT bin
      int ctbNumBinsCot;
      //number of used zT bins inside GP ZT bin
      int ctbNumBinsZT;
      // required number of stub layers to form a candidate
      int ctbMinLayers;
      // max number of output tracks per node
      int ctbMaxTracks;
      // cut on number of stub per layer for input candidates
      int ctbMaxStubs;
      // internal memory depth
      int ctbDepthMemory;
      // number of kf worker
      int kfNumWorker;
      // max number of tracks a kf worker can process
      int kfMaxTracks;
      // required number of stub layers to form a track
      int kfMinLayers;
      // maximum number of  layers added to a track
      int kfMaxLayers;
      // maximum number of layer gaps allowed during cominatorical track building
      int kfMaxGaps;
      // perform seeding in layers 0 to this
      int kfMaxSeedingLayer;
      // number of stubs forming a seed
      int kfNumSeedStubs;
      // shiting chi2 in r-phi plane by power of two when caliclating chi2
      int kfShiftChi20;
      // shiting chi2 in r-z plane by power of two when caliclating chi2
      int kfShiftChi21;
      // cut on chi2 over degree of freedom
      double kfCutChi2;
      // number of bits used to represent chi2 over degree of freedom
      int kfWidthChi2;
      // internal memory depth
      int drDepthMemory;
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
      // f/w bin edge integer values to bin mva
      std::vector<int> tqBinEdges;
    };
    Setup() {}
    Setup(const Config& iConfig, const trackerDTC::Setup* setup);
    ~Setup() = default;

    // stub layer id (barrel: 1 - 6, endcap: 11 - 15)
    int layerId(const TTStubRef& ttStubRef) const {
      const DetId& detId = ttStubRef->getDetId();
      return detId.subdetId() == Phase2Tracker::Subdetector::Barrel ? trackerTopology()->layer(detId)
                                                                    : trackerTopology()->endcapWheelP2(detId) + 10;
    }
    // sensor module for ttStubRef
    const trackerDTC::SensorModule* sensorModule(const TTStubRef& ttStubRef) const {
      return dtc_->sensorModule(ttStubRef);
    }
    // collection of outer tracker sensor modules
    const std::vector<trackerDTC::SensorModule>& sensorModules() const { return dtc_->sensorModules(); }
    // TrackerGeometry
    const TrackerGeometry* trackerGeometry() const { return dtc_->trackerGeometry(); }
    // TrackerTopology
    const TrackerTopology* trackerTopology() const { return dtc_->trackerTopology(); }

    // number of detector layer a particle may cross
    int sysNumLayer() const { return dtc_->sysNumLayer(); }
    // pixel pitch of outer tracker PS sensors in cm
    double mpaPitch() const { return dtc_->mpaPitch(); }
    //strip pitch of outer tracker 2S sensors in cm
    double cbcPitch() const { return dtc_->cbcPitch(); }
    // pixel length of outer tracker PS sensors in cm
    double mpaLength() const { return dtc_->mpaLength(); }
    // strip length of outer tracker 2S sensors in cm
    double cbcLength() const { return dtc_->cbcLength(); }
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double smTiltUncertaintyR() const { return dtc_->smTiltUncertaintyR(); }
    // critical radius in cm defining r-phi Region shape
    double regChosenRofPhi() const { return dtc_->regChosenRofPhi(); }
    // additional radial uncertainty in cm used to calculate stub phi residual uncertainty to take multiple scattering into account
    double smScattering() const { return dtc_->smScattering(); }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxSlope() const { return dtc_->smTiltApproxSlope(); }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double smTiltApproxIntercept() const { return dtc_->smTiltApproxIntercept(); }
    // critical radius in cm defining r-z Region shape
    double regChosenRofZ() const { return dtc_->regChosenRofZ(); }
    // enables simulation of truncation
    bool enableTruncation() const { return dtc_->enableTruncation(); }
    // number of frames which can be processed internally using high clock frequency
    int numFrames() const { return dtc_->sysNumFrames() + dtc_->sysNumFramesInfra() - 1; }
    // in T
    double sysBField() const { return dtc_->sysBField(); }
    // number of DTC boards used to readout a detector region
    int regNumDTC() const { return dtc_->regNumDTC(); }
    // number of DTC boards connected to one TFP
    int numDTCsPerTFP() const { return dtc_->regNumDTC() * dtc_->sysNumOverlap(); }
    // total number of sectors
    int numSectors() const { return gpNumSector_; }
    // number of phi slices the outer tracker readout is organized in
    int sysNumRegion() const { return dtc_->sysNumRegion(); }
    // number of bits used for stub r - ChosenRofPhi
    int glWidthR() const { return dtc_->glWidthR(); }
    // number of bits used for stub phi w.r.t. phi region centre
    int glWidthPhi() const { return dtc_->glWidthPhi(); }
    // number of bits used for stub z
    int glWidthZ() const { return dtc_->glWidthZ(); }
    // largest possible |r - chosenRofPhi|
    double maxRphi() const { return maxRphi_; }
    // half length of outer tracker in cm
    double sysHalfLength() const { return dtc_->sysHalfLength(); }
    // half lumi region size in cm defining r-z Region shape
    double regBeamWindowZ() const { return dtc_->regBeamWindowZ(); }
    // largest possible |r - chosenRofZ|
    double maxRz() const { return maxRz_; }
    // defining r-z Region shape in cm
    double regMaxEta() const { return dtc_->regMaxEta(); }
    // converts GeV in 1/cm
    double invPtToDphi() const { return dtc_->sysInvPtToDphi(); }
    // min pt in GeV used in stub format
    double minPt() const { return dtc_->stubMinPt(); }
    // inner radius of outer tracker in cm
    double sysInnerRadius() const { return dtc_->sysInnerRadius(); }
    // number of frames which can be send out of DTC/TFP per event
    int sysNumFrames() const { return dtc_->sysNumFrames(); }
    // needed gap between events of emp-infrastructure firmware
    int numFramesInfra() const { return dtc_->sysNumFramesInfra(); }
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
    // mean z of outer tracker endcap disks
    double stubDiskZ(int disk) const { return dtc_->stubDiskZ(disk); }
    // mean r of outer tracker endcap disk
    double stubDiskR(int disk) const { return dtc_->stubDiskR(disk, 0); }
    // single region phiT range in rad [2pi/9]
    double regRangePhiT() const { return dtc_->regRangePhiT(); }

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
    // number of phi sectors in a processing nonant used in hough transform
    int gpNumBinsPhiT() const { return config_.gpNumBinsPhiT; }
    // number of eta sectors used in hough transform
    int gpNumBinsZT() const { return config_.gpNumBinsZT; }
    // fifo depth in stub router firmware
    int gpDepthMemory() const { return config_.gpDepthMemory; }
    // phi sector size in rad
    double gpRangePhiT() const { return gpRangePhiT_; }
    // total number of sectors
    int gpNumSector() const { return gpNumSector_; }
    // number of inv2R bins used in hough transform
    int htNumBinsInv2R() const { return config_.htNumBinsInv2R; }
    // number of phiT bins used in hough transform
    int htNumBinsPhiT() const { return config_.htNumBinsPhiT; }
    // required number of stub layers to form a candidate
    int htMinLayers() const { return config_.htMinLayers; }
    // internal fifo depth
    int htDepthMemory() const { return config_.htDepthMemory; }
    // number of finer inv2R bins inside HT bin
    int ctbNumBinsInv2R() const { return config_.ctbNumBinsInv2R; }
    // number of finer phiT bins inside HT bin
    int ctbNumBinsPhiT() const { return config_.ctbNumBinsPhiT; }
    // number of used z0 bins inside GP ZT bin
    int ctbNumBinsCot() const { return config_.ctbNumBinsCot; }
    //number of used zT bins inside GP ZT bin
    int ctbNumBinsZT() const { return config_.ctbNumBinsZT; }
    // required number of stub layers to form a candidate
    int ctbMinLayers() const { return config_.ctbMinLayers; }
    // max number of output tracks per node
    int ctbMaxTracks() const { return config_.ctbMaxTracks; }
    // cut on number of stub per layer for input candidates
    int ctbMaxStubs() const { return config_.ctbMaxStubs; }
    // internal memory depth
    int ctbDepthMemory() const { return config_.ctbDepthMemory; }
    // number of kf worker
    int kfNumWorker() const { return config_.kfNumWorker; }
    // max number of tracks a kf worker can process
    int kfMaxTracks() const { return config_.kfMaxTracks; }
    // required number of stub layers to form a track
    int kfMinLayers() const { return config_.kfMinLayers; }
    // maximum number of  layers added to a track
    int kfMaxLayers() const { return config_.kfMaxLayers; }
    // maximum number of layer gaps allowed during cominatorical track building
    int kfMaxGaps() const { return config_.kfMaxGaps; }
    // perform seeding in layers 0 to this
    int kfMaxSeedingLayer() const { return config_.kfMaxSeedingLayer; }
    // number of stubs forming a seed
    int kfNumSeedStubs() const { return config_.kfNumSeedStubs; }
    // shiting chi2 in r-phi plane by power of two when caliclating chi2
    int kfShiftChi20() const { return config_.kfShiftChi20; }
    // shiting chi2 in r-z plane by power of two when caliclating chi2
    int kfShiftChi21() const { return config_.kfShiftChi21; }
    // cut on chi2 over degree of freedom
    double kfCutChi2() const { return config_.kfCutChi2; }
    // number of bits used to represent chi2 over degree of freedom
    int kfWidthChi2() const { return config_.kfWidthChi2; }
    // internal memory depth
    int drDepthMemory() const { return config_.drDepthMemory; }
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
    // f/w bin edge integer values to bin mva
    const std::vector<int>& tqBinEdges() const { return config_.tqBinEdges; }

  private:
    // DTC setup
    const trackerDTC::Setup* dtc_;
    // Configuration
    Config config_;
    // phi sector size in rad
    double gpRangePhiT_;
    // total number of sectors
    int gpNumSector_;
    // number of bits used to count stubs per layer
    int ctbWidthLayerCount_;
    // largest possible |r - chosenRofPhi|
    double maxRphi_;
    // largest possible |r - chosenRofZ|
    double maxRz_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::Setup, trackerDTC::SetupRcd);

#endif
