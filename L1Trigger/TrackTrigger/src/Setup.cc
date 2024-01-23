#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace tt {

  Setup::Setup(const ParameterSet& iConfig,
               const MagneticField& magneticField,
               const TrackerGeometry& trackerGeometry,
               const TrackerTopology& trackerTopology,
               const TrackerDetToDTCELinkCablingMap& cablingMap,
               const StubAlgorithmOfficial& stubAlgorithm,
               const ParameterSet& pSetStubAlgorithm,
               const ParameterSet& pSetGeometryConfiguration,
               const ParameterSetID& pSetIdTTStubAlgorithm,
               const ParameterSetID& pSetIdGeometryConfiguration)
      : magneticField_(&magneticField),
        trackerGeometry_(&trackerGeometry),
        trackerTopology_(&trackerTopology),
        cablingMap_(&cablingMap),
        stubAlgorithm_(&stubAlgorithm),
        pSetSA_(&pSetStubAlgorithm),
        pSetGC_(&pSetGeometryConfiguration),
        pSetIdTTStubAlgorithm_(pSetIdTTStubAlgorithm),
        pSetIdGeometryConfiguration_(pSetIdGeometryConfiguration),
        // DD4hep
        fromDD4hep_(iConfig.getParameter<bool>("fromDD4hep")),
        // Parameter to check if configured Tracker Geometry is supported
        pSetSG_(iConfig.getParameter<ParameterSet>("UnSupportedGeometry")),
        sgXMLLabel_(pSetSG_.getParameter<string>("XMLLabel")),
        sgXMLPath_(pSetSG_.getParameter<string>("XMLPath")),
        sgXMLFile_(pSetSG_.getParameter<string>("XMLFile")),
        sgXMLVersions_(pSetSG_.getParameter<vector<string>>("XMLVersions")),
        // Parameter to check if Process History is consistent with process configuration
        pSetPH_(iConfig.getParameter<ParameterSet>("ProcessHistory")),
        phGeometryConfiguration_(pSetPH_.getParameter<string>("GeometryConfiguration")),
        phTTStubAlgorithm_(pSetPH_.getParameter<string>("TTStubAlgorithm")),
        // Common track finding parameter
        pSetTF_(iConfig.getParameter<ParameterSet>("TrackFinding")),
        beamWindowZ_(pSetTF_.getParameter<double>("BeamWindowZ")),
        matchedLayers_(pSetTF_.getParameter<int>("MatchedLayers")),
        matchedLayersPS_(pSetTF_.getParameter<int>("MatchedLayersPS")),
        unMatchedStubs_(pSetTF_.getParameter<int>("UnMatchedStubs")),
        unMatchedStubsPS_(pSetTF_.getParameter<int>("UnMatchedStubsPS")),
        scattering_(pSetTF_.getParameter<double>("Scattering")),
        // TMTT specific parameter
        pSetTMTT_(iConfig.getParameter<ParameterSet>("TMTT")),
        minPt_(pSetTMTT_.getParameter<double>("MinPt")),
        maxEta_(pSetTMTT_.getParameter<double>("MaxEta")),
        chosenRofPhi_(pSetTMTT_.getParameter<double>("ChosenRofPhi")),
        numLayers_(pSetTMTT_.getParameter<int>("NumLayers")),
        tmttWidthR_(pSetTMTT_.getParameter<int>("WidthR")),
        tmttWidthPhi_(pSetTMTT_.getParameter<int>("WidthPhi")),
        tmttWidthZ_(pSetTMTT_.getParameter<int>("WidthZ")),
        // Hybrid specific parameter
        pSetHybrid_(iConfig.getParameter<ParameterSet>("Hybrid")),
        hybridMinPtStub_(pSetHybrid_.getParameter<double>("MinPtStub")),
        hybridMinPtCand_(pSetHybrid_.getParameter<double>("MinPtCand")),
        hybridMaxEta_(pSetHybrid_.getParameter<double>("MaxEta")),
        hybridChosenRofPhi_(pSetHybrid_.getParameter<double>("ChosenRofPhi")),
        hybridNumLayers_(pSetHybrid_.getParameter<int>("NumLayers")),
        hybridNumRingsPS_(pSetHybrid_.getParameter<vector<int>>("NumRingsPS")),
        hybridWidthsR_(pSetHybrid_.getParameter<vector<int>>("WidthsR")),
        hybridWidthsZ_(pSetHybrid_.getParameter<vector<int>>("WidthsZ")),
        hybridWidthsPhi_(pSetHybrid_.getParameter<vector<int>>("WidthsPhi")),
        hybridWidthsAlpha_(pSetHybrid_.getParameter<vector<int>>("WidthsAlpha")),
        hybridWidthsBend_(pSetHybrid_.getParameter<vector<int>>("WidthsBend")),
        hybridRangesR_(pSetHybrid_.getParameter<vector<double>>("RangesR")),
        hybridRangesZ_(pSetHybrid_.getParameter<vector<double>>("RangesZ")),
        hybridRangesAlpha_(pSetHybrid_.getParameter<vector<double>>("RangesAlpha")),
        hybridLayerRs_(pSetHybrid_.getParameter<vector<double>>("LayerRs")),
        hybridDiskZs_(pSetHybrid_.getParameter<vector<double>>("DiskZs")),
        hybridDisk2SRsSet_(pSetHybrid_.getParameter<vector<ParameterSet>>("Disk2SRsSet")),
        tbInnerRadius_(pSetHybrid_.getParameter<double>("InnerRadius")),
        tbWidthsR_(pSetHybrid_.getParameter<vector<int>>("WidthsRTB")),
        // Parameter specifying TrackingParticle used for Efficiency measurements
        pSetTP_(iConfig.getParameter<ParameterSet>("TrackingParticle")),
        tpMinPt_(pSetTP_.getParameter<double>("MinPt")),
        tpMaxEta_(pSetTP_.getParameter<double>("MaxEta")),
        tpMaxVertR_(pSetTP_.getParameter<double>("MaxVertR")),
        tpMaxVertZ_(pSetTP_.getParameter<double>("MaxVertZ")),
        tpMaxD0_(pSetTP_.getParameter<double>("MaxD0")),
        tpMinLayers_(pSetTP_.getParameter<int>("MinLayers")),
        tpMinLayersPS_(pSetTP_.getParameter<int>("MinLayersPS")),
        tpMaxBadStubs2S_(pSetTP_.getParameter<int>("MaxBadStubs2S")),
        tpMaxBadStubsPS_(pSetTP_.getParameter<int>("MaxBadStubsPS")),
        // Fimrware specific Parameter
        pSetFW_(iConfig.getParameter<ParameterSet>("Firmware")),
        widthDSPa_(pSetFW_.getParameter<int>("WidthDSPa")),
        widthDSPb_(pSetFW_.getParameter<int>("WidthDSPb")),
        widthDSPc_(pSetFW_.getParameter<int>("WidthDSPc")),
        widthAddrBRAM36_(pSetFW_.getParameter<int>("WidthAddrBRAM36")),
        widthAddrBRAM18_(pSetFW_.getParameter<int>("WidthAddrBRAM18")),
        numFramesInfra_(pSetFW_.getParameter<int>("NumFramesInfra")),
        freqLHC_(pSetFW_.getParameter<double>("FreqLHC")),
        freqBE_(pSetFW_.getParameter<double>("FreqBE")),
        tmpFE_(pSetFW_.getParameter<int>("TMP_FE")),
        tmpTFP_(pSetFW_.getParameter<int>("TMP_TFP")),
        speedOfLight_(pSetFW_.getParameter<double>("SpeedOfLight")),
        bField_(pSetFW_.getParameter<double>("BField")),
        bFieldError_(pSetFW_.getParameter<double>("BFieldError")),
        outerRadius_(pSetFW_.getParameter<double>("OuterRadius")),
        innerRadius_(pSetFW_.getParameter<double>("InnerRadius")),
        halfLength_(pSetFW_.getParameter<double>("HalfLength")),
        tiltApproxSlope_(pSetFW_.getParameter<double>("TiltApproxSlope")),
        tiltApproxIntercept_(pSetFW_.getParameter<double>("TiltApproxIntercept")),
        tiltUncertaintyR_(pSetFW_.getParameter<double>("TiltUncertaintyR")),
        mindPhi_(pSetFW_.getParameter<double>("MindPhi")),
        maxdPhi_(pSetFW_.getParameter<double>("MaxdPhi")),
        mindZ_(pSetFW_.getParameter<double>("MindZ")),
        maxdZ_(pSetFW_.getParameter<double>("MaxdZ")),
        pitch2S_(pSetFW_.getParameter<double>("Pitch2S")),
        pitchPS_(pSetFW_.getParameter<double>("PitchPS")),
        length2S_(pSetFW_.getParameter<double>("Length2S")),
        lengthPS_(pSetFW_.getParameter<double>("LengthPS")),
        tiltedLayerLimitsZ_(pSetFW_.getParameter<vector<double>>("TiltedLayerLimitsZ")),
        psDiskLimitsR_(pSetFW_.getParameter<vector<double>>("PSDiskLimitsR")),
        // Parmeter specifying front-end
        pSetFE_(iConfig.getParameter<ParameterSet>("FrontEnd")),
        widthBend_(pSetFE_.getParameter<int>("WidthBend")),
        widthCol_(pSetFE_.getParameter<int>("WidthCol")),
        widthRow_(pSetFE_.getParameter<int>("WidthRow")),
        baseBend_(pSetFE_.getParameter<double>("BaseBend")),
        baseCol_(pSetFE_.getParameter<double>("BaseCol")),
        baseRow_(pSetFE_.getParameter<double>("BaseRow")),
        baseWindowSize_(pSetFE_.getParameter<double>("BaseWindowSize")),
        bendCut_(pSetFE_.getParameter<double>("BendCut")),
        // Parmeter specifying DTC
        pSetDTC_(iConfig.getParameter<ParameterSet>("DTC")),
        numRegions_(pSetDTC_.getParameter<int>("NumRegions")),
        numOverlappingRegions_(pSetDTC_.getParameter<int>("NumOverlappingRegions")),
        numATCASlots_(pSetDTC_.getParameter<int>("NumATCASlots")),
        numDTCsPerRegion_(pSetDTC_.getParameter<int>("NumDTCsPerRegion")),
        numModulesPerDTC_(pSetDTC_.getParameter<int>("NumModulesPerDTC")),
        dtcNumRoutingBlocks_(pSetDTC_.getParameter<int>("NumRoutingBlocks")),
        dtcDepthMemory_(pSetDTC_.getParameter<int>("DepthMemory")),
        dtcWidthRowLUT_(pSetDTC_.getParameter<int>("WidthRowLUT")),
        dtcWidthInv2R_(pSetDTC_.getParameter<int>("WidthInv2R")),
        offsetDetIdDSV_(pSetDTC_.getParameter<int>("OffsetDetIdDSV")),
        offsetDetIdTP_(pSetDTC_.getParameter<int>("OffsetDetIdTP")),
        offsetLayerDisks_(pSetDTC_.getParameter<int>("OffsetLayerDisks")),
        offsetLayerId_(pSetDTC_.getParameter<int>("OffsetLayerId")),
        numBarrelLayer_(pSetDTC_.getParameter<int>("NumBarrelLayer")),
        slotLimitPS_(pSetDTC_.getParameter<int>("SlotLimitPS")),
        slotLimit10gbps_(pSetDTC_.getParameter<int>("SlotLimit10gbps")),
        // Parmeter specifying TFP
        pSetTFP_(iConfig.getParameter<ParameterSet>("TFP")),
        tfpWidthPhi0_(pSetTFP_.getParameter<int>("WidthPhi0")),
        tfpWidthInv2R_(pSetTFP_.getParameter<int>("WidthInv2R")),
        tfpWidthCot_(pSetTFP_.getParameter<int>("WidthCot")),
        tfpWidthZ0_(pSetTFP_.getParameter<int>("WidthZ0")),
        tfpNumChannel_(pSetTFP_.getParameter<int>("NumChannel")),
        // Parmeter specifying GeometricProcessor
        pSetGP_(iConfig.getParameter<ParameterSet>("GeometricProcessor")),
        numSectorsPhi_(pSetGP_.getParameter<int>("NumSectorsPhi")),
        chosenRofZ_(pSetGP_.getParameter<double>("ChosenRofZ")),
        neededRangeChiZ_(pSetGP_.getParameter<double>("RangeChiZ")),
        gpDepthMemory_(pSetGP_.getParameter<int>("DepthMemory")),
        boundariesEta_(pSetGP_.getParameter<vector<double>>("BoundariesEta")),
        // Parmeter specifying HoughTransform
        pSetHT_(iConfig.getParameter<ParameterSet>("HoughTransform")),
        htNumBinsInv2R_(pSetHT_.getParameter<int>("NumBinsInv2R")),
        htNumBinsPhiT_(pSetHT_.getParameter<int>("NumBinsPhiT")),
        htMinLayers_(pSetHT_.getParameter<int>("MinLayers")),
        htDepthMemory_(pSetHT_.getParameter<int>("DepthMemory")),
        // Parmeter specifying MiniHoughTransform
        pSetMHT_(iConfig.getParameter<ParameterSet>("MiniHoughTransform")),
        mhtNumBinsInv2R_(pSetMHT_.getParameter<int>("NumBinsInv2R")),
        mhtNumBinsPhiT_(pSetMHT_.getParameter<int>("NumBinsPhiT")),
        mhtNumDLBs_(pSetMHT_.getParameter<int>("NumDLBs")),
        mhtNumDLBNodes_(pSetMHT_.getParameter<int>("NumDLBNodes")),
        mhtNumDLBChannel_(pSetMHT_.getParameter<int>("NumDLBChannel")),
        mhtMinLayers_(pSetMHT_.getParameter<int>("MinLayers")),
        // Parmeter specifying ZHoughTransform
        pSetZHT_(iConfig.getParameter<ParameterSet>("ZHoughTransform")),
        zhtNumBinsZT_(pSetZHT_.getParameter<int>("NumBinsZT")),
        zhtNumBinsCot_(pSetZHT_.getParameter<int>("NumBinsCot")),
        zhtNumStages_(pSetZHT_.getParameter<int>("NumStages")),
        zhtMinLayers_(pSetZHT_.getParameter<int>("MinLayers")),
        zhtMaxTracks_(pSetZHT_.getParameter<int>("MaxTracks")),
        zhtMaxStubsPerLayer_(pSetZHT_.getParameter<int>("MaxStubsPerLayer")),
        // Parameter specifying KalmanFilter Input Formatter
        pSetKFin_(iConfig.getParameter<ParameterSet>("KalmanFilterIn")),
        kfinShiftRangePhi_(pSetKFin_.getParameter<int>("ShiftRangePhi")),
        kfinShiftRangeZ_(pSetKFin_.getParameter<int>("ShiftRangeZ")),
        // Parmeter specifying KalmanFilter
        pSetKF_(iConfig.getParameter<ParameterSet>("KalmanFilter")),
        kfNumWorker_(pSetKF_.getParameter<int>("NumWorker")),
        kfMinLayers_(pSetKF_.getParameter<int>("MinLayers")),
        kfMaxLayers_(pSetKF_.getParameter<int>("MaxLayers")),
        kfRangeFactor_(pSetKF_.getParameter<double>("RangeFactor")),
        kfShiftInitialC00_(pSetKF_.getParameter<int>("ShiftInitialC00")),
        kfShiftInitialC11_(pSetKF_.getParameter<int>("ShiftInitialC11")),
        kfShiftInitialC22_(pSetKF_.getParameter<int>("ShiftInitialC22")),
        kfShiftInitialC33_(pSetKF_.getParameter<int>("ShiftInitialC33")),
        // Parmeter specifying KalmanFilter Output Formatter
        pSetKFOut_(iConfig.getParameter<ParameterSet>("KalmanFilterOut")),
        kfoutchi2rphiConv_(pSetKFOut_.getParameter<int>("Chi2rphiConv")),
        kfoutchi2rzConv_(pSetKFOut_.getParameter<int>("Chi2rzConv")),
        weightBinFraction_(pSetKFOut_.getParameter<int>("WeightBinFraction")),
        dzTruncation_(pSetKFOut_.getParameter<int>("DzTruncation")),
        dphiTruncation_(pSetKFOut_.getParameter<int>("DphiTruncation")),
        // Parmeter specifying DuplicateRemoval
        pSetDR_(iConfig.getParameter<ParameterSet>("DuplicateRemoval")),
        drDepthMemory_(pSetDR_.getParameter<int>("DepthMemory")) {
    configurationSupported_ = true;
    // check if bField is supported
    checkMagneticField();
    // check if geometry is supported
    checkGeometry();
    if (!configurationSupported_)
      return;
    // derive constants
    calculateConstants();
    // convert configuration of TTStubAlgorithm
    consumeStubAlgorithm();
    // create all possible encodingsBend
    encodingsBendPS_.reserve(maxWindowSize_ + 1);
    encodingsBend2S_.reserve(maxWindowSize_ + 1);
    encodeBend(encodingsBendPS_, true);
    encodeBend(encodingsBend2S_, false);
    // create sensor modules
    produceSensorModules();
    // configure TPSelector
    configureTPSelector();
  }

  // checks current configuration vs input sample configuration
  void Setup::checkHistory(const ProcessHistory& processHistory) const {
    const pset::Registry* psetRegistry = pset::Registry::instance();
    // check used TTStubAlgorithm in input producer
    checkHistory(processHistory, psetRegistry, phTTStubAlgorithm_, pSetIdTTStubAlgorithm_);
    // check used GeometryConfiguration in input producer
    checkHistory(processHistory, psetRegistry, phGeometryConfiguration_, pSetIdGeometryConfiguration_);
  }

  // checks consitency between history and current configuration for a specific module
  void Setup::checkHistory(const ProcessHistory& ph,
                           const pset::Registry* pr,
                           const string& label,
                           const ParameterSetID& pSetId) const {
    vector<pair<string, ParameterSet>> pSets;
    pSets.reserve(ph.size());
    for (const ProcessConfiguration& pc : ph) {
      const ParameterSet* pSet = pr->getMapped(pc.parameterSetID());
      if (pSet && pSet->exists(label))
        pSets.emplace_back(pc.processName(), pSet->getParameterSet(label));
    }
    if (pSets.empty()) {
      cms::Exception exception("BadConfiguration");
      exception << label << " not found in process history.";
      exception.addContext("tt::Setup::checkHistory");
      throw exception;
    }
    auto consistent = [&pSetId](const pair<string, ParameterSet>& p) { return p.second.id() == pSetId; };
    if (!all_of(pSets.begin(), pSets.end(), consistent)) {
      const ParameterSet& pSetProcess = getParameterSet(pSetId);
      cms::Exception exception("BadConfiguration");
      exception.addContext("tt::Setup::checkHistory");
      exception << label << " inconsistent with History." << endl;
      exception << "Current Configuration:" << endl << pSetProcess.dump() << endl;
      for (const pair<string, ParameterSet>& p : pSets)
        if (!consistent(p))
          exception << "Process " << p.first << " Configuration:" << endl << dumpDiff(p.second, pSetProcess) << endl;
      throw exception;
    }
  }

  // dumps pSetHistory where incosistent lines with pSetProcess are highlighted
  string Setup::dumpDiff(const ParameterSet& pSetHistory, const ParameterSet& pSetProcess) const {
    stringstream ssHistory, ssProcess, ss;
    ssHistory << pSetHistory.dump();
    ssProcess << pSetProcess.dump();
    string lineHistory, lineProcess;
    for (; getline(ssHistory, lineHistory) && getline(ssProcess, lineProcess);)
      ss << (lineHistory != lineProcess ? "\033[1;31m" : "") << lineHistory << "\033[0m" << endl;
    return ss.str();
  }

  // converts tk layout id into dtc id
  int Setup::dtcId(int tkLayoutId) const {
    checkTKLayoutId(tkLayoutId);
    const int tkId = tkLayoutId - 1;
    const int side = tkId / (numRegions_ * numATCASlots_);
    const int region = (tkId % (numRegions_ * numATCASlots_)) / numATCASlots_;
    const int slot = tkId % numATCASlots_;
    return region * numDTCsPerRegion_ + side * numATCASlots_ + slot;
  }

  // converts dtc id into tk layout id
  int Setup::tkLayoutId(int dtcId) const {
    checkDTCId(dtcId);
    const int slot = dtcId % numATCASlots_;
    const int region = dtcId / numDTCsPerRegion_;
    const int side = (dtcId % numDTCsPerRegion_) / numATCASlots_;
    return (side * numRegions_ + region) * numATCASlots_ + slot + 1;
  }

  // converts TFP identifier (region[0-8], channel[0-47]) into dtc id
  int Setup::dtcId(int tfpRegion, int tfpChannel) const {
    checkTFPIdentifier(tfpRegion, tfpChannel);
    const int dtcChannel = numOverlappingRegions_ - (tfpChannel / numDTCsPerRegion_) - 1;
    const int dtcBoard = tfpChannel % numDTCsPerRegion_;
    const int dtcRegion = tfpRegion - dtcChannel >= 0 ? tfpRegion - dtcChannel : tfpRegion - dtcChannel + numRegions_;
    return dtcRegion * numDTCsPerRegion_ + dtcBoard;
  }

  // checks if given DTC id is connected to PS or 2S sensormodules
  bool Setup::psModule(int dtcId) const {
    checkDTCId(dtcId);
    // from tklayout: first 3 are 10 gbps PS, next 3 are 5 gbps PS and residual 6 are 5 gbps 2S modules
    return slot(dtcId) < slotLimitPS_;
  }

  // return sensor moduel type
  SensorModule::Type Setup::type(const TTStubRef& ttStubRef) const {
    const bool barrel = this->barrel(ttStubRef);
    const bool psModule = this->psModule(ttStubRef);
    SensorModule::Type type;
    if (barrel && psModule)
      type = SensorModule::BarrelPS;
    if (barrel && !psModule)
      type = SensorModule::Barrel2S;
    if (!barrel && psModule)
      type = SensorModule::DiskPS;
    if (!barrel && !psModule)
      type = SensorModule::Disk2S;
    return type;
  }

  // checks if given dtcId is connected via 10 gbps link
  bool Setup::gbps10(int dtcId) const {
    checkDTCId(dtcId);
    return slot(dtcId) < slotLimit10gbps_;
  }

  // checks if given dtcId is connected to -z (false) or +z (true)
  bool Setup::side(int dtcId) const {
    checkDTCId(dtcId);
    const int side = (dtcId % numDTCsPerRegion_) / numATCASlots_;
    // from tkLayout: first 12 +z, next 12 -z
    return side == 0;
  }

  // ATCA slot number [0-11] of given dtcId
  int Setup::slot(int dtcId) const {
    checkDTCId(dtcId);
    return dtcId % numATCASlots_;
  }

  // sensor module for det id
  SensorModule* Setup::sensorModule(const DetId& detId) const {
    const auto it = detIdToSensorModule_.find(detId);
    if (it == detIdToSensorModule_.end()) {
      cms::Exception exception("NullPtr");
      exception << "Unknown DetId used.";
      exception.addContext("tt::Setup::sensorModule");
      throw exception;
    }
    return it->second;
  }

  // index = encoded bend, value = decoded bend for given window size and module type
  const vector<double>& Setup::encodingBend(int windowSize, bool psModule) const {
    const vector<vector<double>>& encodingsBend = psModule ? encodingsBendPS_ : encodingsBend2S_;
    return encodingsBend.at(windowSize);
  }

  // check if bField is supported
  void Setup::checkMagneticField() {
    const double bFieldES = magneticField_->inTesla(GlobalPoint(0., 0., 0.)).z();
    if (abs(bField_ - bFieldES) > bFieldError_) {
      configurationSupported_ = false;
      LogWarning("ConfigurationNotSupported")
          << "Magnetic Field from EventSetup (" << bFieldES << ") differs more then " << bFieldError_
          << " from supported value (" << bField_ << "). ";
    }
  }

  // check if geometry is supported
  void Setup::checkGeometry() {
    //FIX ME: Can we assume that geometry used in dd4hep wf supports L1Track?
    if (!fromDD4hep_) {
      const vector<string>& geomXMLFiles = pSetGC_->getParameter<vector<string>>(sgXMLLabel_);
      string version;
      for (const string& geomXMLFile : geomXMLFiles) {
        const auto begin = geomXMLFile.find(sgXMLPath_) + sgXMLPath_.size();
        const auto end = geomXMLFile.find(sgXMLFile_);
        if (begin != string::npos && end != string::npos)
          version = geomXMLFile.substr(begin, end - begin - 1);
      }
      if (version.empty()) {
        cms::Exception exception("LogicError");
        exception << "No " << sgXMLPath_ << "*/" << sgXMLFile_ << " found in GeometryConfiguration";
        exception.addContext("tt::Setup::checkGeometry");
        throw exception;
      }
      if (find(sgXMLVersions_.begin(), sgXMLVersions_.end(), version) != sgXMLVersions_.end()) {
        configurationSupported_ = false;
        LogWarning("ConfigurationNotSupported")
            << "Geometry Configuration " << sgXMLPath_ << version << "/" << sgXMLFile_ << " is not supported. ";
      }
    }
  }

  // convert configuration of TTStubAlgorithm
  void Setup::consumeStubAlgorithm() {
    numTiltedLayerRings_ = pSetSA_->getParameter<vector<double>>("NTiltedRings");
    windowSizeBarrelLayers_ = pSetSA_->getParameter<vector<double>>("BarrelCut");
    const auto& pSetsTiltedLayer = pSetSA_->getParameter<vector<ParameterSet>>("TiltedBarrelCutSet");
    const auto& pSetsEncapDisks = pSetSA_->getParameter<vector<ParameterSet>>("EndcapCutSet");
    windowSizeTiltedLayerRings_.reserve(pSetsTiltedLayer.size());
    for (const auto& pSet : pSetsTiltedLayer)
      windowSizeTiltedLayerRings_.emplace_back(pSet.getParameter<vector<double>>("TiltedCut"));
    windowSizeEndcapDisksRings_.reserve(pSetsEncapDisks.size());
    for (const auto& pSet : pSetsEncapDisks)
      windowSizeEndcapDisksRings_.emplace_back(pSet.getParameter<vector<double>>("EndcapCut"));
    maxWindowSize_ = -1;
    for (const auto& windowss : {windowSizeTiltedLayerRings_, windowSizeEndcapDisksRings_, {windowSizeBarrelLayers_}})
      for (const auto& windows : windowss)
        for (const auto& window : windows)
          maxWindowSize_ = max(maxWindowSize_, (int)(window / baseWindowSize_));
  }

  // create bend encodings
  void Setup::encodeBend(vector<vector<double>>& encodings, bool ps) const {
    for (int window = 0; window < maxWindowSize_ + 1; window++) {
      set<double> encoding;
      for (int bend = 0; bend < window + 1; bend++)
        encoding.insert(stubAlgorithm_->degradeBend(ps, window, bend));
      encodings.emplace_back(encoding.begin(), encoding.end());
    }
  }

  // create sensor modules
  void Setup::produceSensorModules() {
    sensorModules_.reserve(numModules_);
    dtcModules_ = vector<vector<SensorModule*>>(numDTCs_);
    for (vector<SensorModule*>& dtcModules : dtcModules_)
      dtcModules.reserve(numModulesPerDTC_);
    enum SubDetId { pixelBarrel = 1, pixelDisks = 2 };
    // loop over all tracker modules
    for (const DetId& detId : trackerGeometry_->detIds()) {
      // skip pixel detector
      if (detId.subdetId() == pixelBarrel || detId.subdetId() == pixelDisks)
        continue;
      // skip multiple detIds per module
      if (!trackerTopology_->isLower(detId))
        continue;
      // lowerDetId - 1 = tk layout det id
      const DetId detIdTkLayout = detId + offsetDetIdTP_;
      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = cablingMap_->detIdToDTCELinkId(detIdTkLayout).first->second.dtc_id();
      // track trigger dtc id [0-215]
      const int dtcId = Setup::dtcId(tklId);
      // collection of so far connected modules to this dtc
      vector<SensorModule*>& dtcModules = dtcModules_[dtcId];
      // construct sendor module
      sensorModules_.emplace_back(this, detId, dtcId, dtcModules.size());
      SensorModule* sensorModule = &sensorModules_.back();
      // store connection between detId and sensor module
      detIdToSensorModule_.emplace(detId, sensorModule);
      // store connection between dtcId and sensor module
      dtcModules.push_back(sensorModule);
    }
    for (vector<SensorModule*>& dtcModules : dtcModules_) {
      dtcModules.shrink_to_fit();
      // check configuration
      if ((int)dtcModules.size() > numModulesPerDTC_) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << numModulesPerDTC_ << " modules to a DTC.";
        exception.addContext("tt::Setup::Setup");
        throw exception;
      }
    }
  }

  // configure TPSelector
  void Setup::configureTPSelector() {
    // configure TrackingParticleSelector
    const double ptMin = tpMinPt_;
    constexpr double ptMax = 9.e9;
    const double etaMax = tpMaxEta_;
    const double tip = tpMaxVertR_;
    const double lip = tpMaxVertZ_;
    constexpr int minHit = 0;
    constexpr bool signalOnly = true;
    constexpr bool intimeOnly = true;
    constexpr bool chargedOnly = true;
    constexpr bool stableOnly = false;
    tpSelector_ = TrackingParticleSelector(
        ptMin, ptMax, -etaMax, etaMax, tip, lip, minHit, signalOnly, intimeOnly, chargedOnly, stableOnly);
    tpSelectorLoose_ =
        TrackingParticleSelector(ptMin, ptMax, -etaMax, etaMax, tip, lip, minHit, false, false, false, stableOnly);
  }

  // stub layer id (barrel: 1 - 6, endcap: 11 - 15)
  int Setup::layerId(const TTStubRef& ttStubRef) const {
    const DetId& detId = ttStubRef->getDetId();
    return detId.subdetId() == StripSubdetector::TOB ? trackerTopology_->layer(detId)
                                                     : trackerTopology_->tidWheel(detId) + offsetLayerDisks_;
  }

  // return tracklet layerId (barrel: [0-5], endcap: [6-10]) for given TTStubRef
  int Setup::trackletLayerId(const TTStubRef& ttStubRef) const {
    return this->layerId(ttStubRef) - (this->barrel(ttStubRef) ? offsetLayerId_ : numBarrelLayer_ - offsetLayerId_);
  }

  // return index layerId (barrel: [0-5], endcap: [0-6]) for given TTStubRef
  int Setup::indexLayerId(const TTStubRef& ttStubRef) const {
    return this->layerId(ttStubRef) - (this->barrel(ttStubRef) ? offsetLayerId_ : offsetLayerId_ + offsetLayerDisks_);
  }

  // true if stub from barrel module
  bool Setup::barrel(const TTStubRef& ttStubRef) const {
    const DetId& detId = ttStubRef->getDetId();
    return detId.subdetId() == StripSubdetector::TOB;
  }

  // true if stub from barrel module
  bool Setup::psModule(const TTStubRef& ttStubRef) const {
    const DetId& detId = ttStubRef->getDetId();
    return trackerGeometry_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
  }

  //
  TTBV Setup::layerMap(const vector<int>& ints) const {
    TTBV ttBV;
    for (int layer = numLayers_ - 1; layer >= 0; layer--) {
      const int i = ints[layer];
      ttBV += TTBV(i, kfWidthLayerCount_);
    }
    return ttBV;
  }

  //
  TTBV Setup::layerMap(const TTBV& hitPattern, const vector<int>& ints) const {
    TTBV ttBV;
    for (int layer = numLayers_ - 1; layer >= 0; layer--) {
      const int i = ints[layer];
      ttBV += TTBV((hitPattern[layer] ? i - 1 : 0), kfWidthLayerCount_);
    }
    return ttBV;
  }

  //
  vector<int> Setup::layerMap(const TTBV& hitPattern, const TTBV& ttBV) const {
    TTBV bv(ttBV);
    vector<int> ints(numLayers_, 0);
    for (int layer = 0; layer < numLayers_; layer++) {
      const int i = bv.extract(kfWidthLayerCount_);
      ints[layer] = i + (hitPattern[layer] ? 1 : 0);
    }
    return ints;
  }

  //
  vector<int> Setup::layerMap(const TTBV& ttBV) const {
    TTBV bv(ttBV);
    vector<int> ints(numLayers_, 0);
    for (int layer = 0; layer < numLayers_; layer++)
      ints[layer] = bv.extract(kfWidthLayerCount_);
    return ints;
  }

  // stub projected phi uncertainty
  double Setup::dPhi(const TTStubRef& ttStubRef, double inv2R) const {
    const DetId& detId = ttStubRef->getDetId();
    SensorModule* sm = sensorModule(detId + 1);
    const double r = stubPos(ttStubRef).perp();
    const double sigma = sm->pitchRow() / r;
    const double scat = scattering_ * abs(inv2R);
    const double extra = sm->barrel() ? 0. : sm->pitchCol() * abs(inv2R);
    const double digi = tmttBasePhi_;
    const double dPhi = sigma + scat + extra + digi;
    if (dPhi >= maxdPhi_ || dPhi < mindPhi_) {
      cms::Exception exception("out_of_range");
      exception.addContext("tt::Setup::dPhi");
      exception << "Stub phi uncertainty " << dPhi << " "
                << "is out of range " << mindPhi_ << " to " << maxdPhi_ << ".";
      throw exception;
    }
    return dPhi;
  }

  // stub projected z uncertainty
  double Setup::dZ(const TTStubRef& ttStubRef, double cot) const {
    const DetId& detId = ttStubRef->getDetId();
    SensorModule* sm = sensorModule(detId + 1);
    const double sigma = sm->pitchCol() * sm->tiltCorrection(cot);
    const double digi = tmttBaseZ_;
    const double dZ = sigma + digi;
    if (dZ >= maxdZ_ || dZ < mindZ_) {
      cms::Exception exception("out_of_range");
      exception.addContext("tt::Setup::dZ");
      exception << "Stub z uncertainty " << dZ << " "
                << "is out of range " << mindZ_ << " to " << maxdZ_ << ".";
      throw exception;
    }
    return dZ;
  }

  // stub projected chi2phi wheight
  double Setup::v0(const TTStubRef& ttStubRef, double inv2R) const {
    const DetId& detId = ttStubRef->getDetId();
    SensorModule* sm = sensorModule(detId + 1);
    const double r = stubPos(ttStubRef).perp();
    const double sigma = pow(sm->pitchRow() / r, 2) / 12.;
    const double scat = pow(scattering_ * inv2R, 2);
    const double extra = sm->barrel() ? 0. : pow(sm->pitchCol() * inv2R, 2);
    const double digi = pow(tmttBasePhi_ / 12., 2);
    return sigma + scat + extra + digi;
  }

  // stub projected chi2z wheight
  double Setup::v1(const TTStubRef& ttStubRef, double cot) const {
    const DetId& detId = ttStubRef->getDetId();
    SensorModule* sm = sensorModule(detId + 1);
    const double sigma = pow(sm->pitchCol() * sm->tiltCorrection(cot), 2) / 12.;
    const double digi = pow(tmttBaseZ_ / 12., 2);
    return sigma + digi;
  }

  // checks if stub collection is considered forming a reconstructable track
  bool Setup::reconstructable(const vector<TTStubRef>& ttStubRefs) const {
    set<int> hitPattern;
    for (const TTStubRef& ttStubRef : ttStubRefs)
      hitPattern.insert(layerId(ttStubRef));
    return (int)hitPattern.size() >= tpMinLayers_;
  }

  // checks if tracking particle is selected for efficiency measurements
  bool Setup::useForAlgEff(const TrackingParticle& tp) const {
    const bool selected = tpSelector_(tp);
    const double cot = sinh(tp.eta());
    const double s = sin(tp.phi());
    const double c = cos(tp.phi());
    const TrackingParticle::Point& v = tp.vertex();
    const double z0 = v.z() - (v.x() * c + v.y() * s) * cot;
    const double d0 = v.x() * s - v.y() * c;
    return selected && (abs(d0) < tpMaxD0_) && (abs(z0) < tpMaxVertZ_);
  }

  // derive constants
  void Setup::calculateConstants() {
    // emp
    const int numFramesPerBX = freqBE_ / freqLHC_;
    numFrames_ = numFramesPerBX * tmpTFP_ - 1;
    numFramesIO_ = numFramesPerBX * tmpTFP_ - numFramesInfra_;
    numFramesFE_ = numFramesPerBX * tmpFE_ - numFramesInfra_;
    // dsp
    widthDSPab_ = widthDSPa_ - 1;
    widthDSPau_ = widthDSPab_ - 1;
    widthDSPbb_ = widthDSPb_ - 1;
    widthDSPbu_ = widthDSPbb_ - 1;
    widthDSPcb_ = widthDSPc_ - 1;
    widthDSPcu_ = widthDSPcb_ - 1;
    // firmware
    maxPitch_ = max(pitchPS_, pitch2S_);
    maxLength_ = max(lengthPS_, length2S_);
    // common track finding
    invPtToDphi_ = speedOfLight_ * bField_ / 2000.;
    baseRegion_ = 2. * M_PI / numRegions_;
    // gp
    baseSector_ = baseRegion_ / numSectorsPhi_;
    maxCot_ = sinh(maxEta_);
    maxZT_ = maxCot_ * chosenRofZ_;
    numSectorsEta_ = boundariesEta_.size() - 1;
    numSectors_ = numSectorsPhi_ * numSectorsEta_;
    sectorCots_.reserve(numSectorsEta_);
    for (int eta = 0; eta < numSectorsEta_; eta++)
      sectorCots_.emplace_back((sinh(boundariesEta_.at(eta)) + sinh(boundariesEta_.at(eta + 1))) / 2.);
    // tmtt
    const double rangeInv2R = 2. * invPtToDphi_ / minPt_;
    tmttBaseInv2R_ = rangeInv2R / htNumBinsInv2R_;
    tmttBasePhiT_ = baseSector_ / htNumBinsPhiT_;
    const double baseRgen = tmttBasePhiT_ / tmttBaseInv2R_;
    const double rangeR = 2. * max(abs(outerRadius_ - chosenRofPhi_), abs(innerRadius_ - chosenRofPhi_));
    const int baseShiftR = ceil(log2(rangeR / baseRgen / pow(2., tmttWidthR_)));
    tmttBaseR_ = baseRgen * pow(2., baseShiftR);
    const double rangeZ = 2. * halfLength_;
    const int baseShiftZ = ceil(log2(rangeZ / tmttBaseR_ / pow(2., tmttWidthZ_)));
    tmttBaseZ_ = tmttBaseR_ * pow(2., baseShiftZ);
    const double rangePhi = baseRegion_ + rangeInv2R * rangeR / 2.;
    const int baseShiftPhi = ceil(log2(rangePhi / tmttBasePhiT_ / pow(2., tmttWidthPhi_)));
    tmttBasePhi_ = tmttBasePhiT_ * pow(2., baseShiftPhi);
    tmttWidthLayer_ = ceil(log2(numLayers_));
    tmttWidthSectorEta_ = ceil(log2(numSectorsEta_));
    tmttWidthInv2R_ = ceil(log2(htNumBinsInv2R_));
    tmttNumUnusedBits_ = TTBV::S_ - tmttWidthLayer_ - 2 * tmttWidthSectorEta_ - tmttWidthR_ - tmttWidthPhi_ -
                         tmttWidthZ_ - 2 * tmttWidthInv2R_ - numSectorsPhi_ - 1;
    // hybrid
    const double hybridRangeInv2R = 2. * invPtToDphi_ / hybridMinPtStub_;
    const double hybridRangeR =
        2. * max(abs(outerRadius_ - hybridChosenRofPhi_), abs(innerRadius_ - hybridChosenRofPhi_));
    hybridRangePhi_ = baseRegion_ + (hybridRangeR * hybridRangeInv2R) / 2.;
    hybridWidthLayerId_ = ceil(log2(hybridNumLayers_));
    hybridBasesZ_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesZ_.emplace_back(hybridRangesZ_.at(type) / pow(2., hybridWidthsZ_.at(type)));
    hybridBasesR_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesR_.emplace_back(hybridRangesR_.at(type) / pow(2., hybridWidthsR_.at(type)));
    hybridBasesR_[SensorModule::Disk2S] = 1.;
    hybridBasesPhi_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesPhi_.emplace_back(hybridRangePhi_ / pow(2., hybridWidthsPhi_.at(type)));
    hybridBasesAlpha_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesAlpha_.emplace_back(hybridRangesAlpha_.at(type) / pow(2., hybridWidthsAlpha_.at(type)));
    hybridNumsUnusedBits_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridNumsUnusedBits_.emplace_back(TTBV::S_ - hybridWidthsR_.at(type) - hybridWidthsZ_.at(type) -
                                         hybridWidthsPhi_.at(type) - hybridWidthsAlpha_.at(type) -
                                         hybridWidthsBend_.at(type) - hybridWidthLayerId_ - 1);
    hybridMaxCot_ = sinh(hybridMaxEta_);
    disk2SRs_.reserve(hybridDisk2SRsSet_.size());
    for (const auto& pSet : hybridDisk2SRsSet_)
      disk2SRs_.emplace_back(pSet.getParameter<vector<double>>("Disk2SRs"));
    // dtc
    numDTCs_ = numRegions_ * numDTCsPerRegion_;
    numDTCsPerTFP_ = numDTCsPerRegion_ * numOverlappingRegions_;
    numModules_ = numDTCs_ * numModulesPerDTC_;
    dtcNumModulesPerRoutingBlock_ = numModulesPerDTC_ / dtcNumRoutingBlocks_;
    dtcNumMergedRows_ = pow(2, widthRow_ - dtcWidthRowLUT_);
    const double maxRangeInv2R = max(rangeInv2R, hybridRangeInv2R);
    const int baseShiftInv2R = ceil(log2(htNumBinsInv2R_)) - dtcWidthInv2R_ + ceil(log2(maxRangeInv2R / rangeInv2R));
    dtcBaseInv2R_ = tmttBaseInv2R_ * pow(2., baseShiftInv2R);
    const int baseDiffM = dtcWidthRowLUT_ - widthRow_;
    dtcBaseM_ = tmttBasePhi_ * pow(2., baseDiffM);
    const double x1 = pow(2, widthRow_) * baseRow_ * maxPitch_ / 2.;
    const double x0 = x1 - pow(2, dtcWidthRowLUT_) * baseRow_ * maxPitch_;
    const double maxM = atan2(x1, innerRadius_) - atan2(x0, innerRadius_);
    dtcWidthM_ = ceil(log2(maxM / dtcBaseM_));
    dtcNumStreams_ = numDTCs_ * numOverlappingRegions_;
    // mht
    mhtNumCells_ = mhtNumBinsInv2R_ * mhtNumBinsPhiT_;
    // zht
    zhtNumCells_ = zhtNumBinsCot_ * zhtNumBinsZT_;
    //
    kfWidthLayerCount_ = ceil(log2(zhtMaxStubsPerLayer_));
  }

  // returns bit accurate hybrid stub radius for given TTStubRef and h/w bit word
  double Setup::stubR(const TTBV& hw, const TTStubRef& ttStubRef) const {
    const bool barrel = this->barrel(ttStubRef);
    const int layerId = this->indexLayerId(ttStubRef);
    const SensorModule::Type type = this->type(ttStubRef);
    const int widthR = hybridWidthsR_.at(type);
    const double baseR = hybridBasesR_.at(type);
    const TTBV hwR(hw, widthR, 0, barrel);
    double r = hwR.val(baseR) + (barrel ? hybridLayerRs_.at(layerId) : 0.);
    if (type == SensorModule::Disk2S)
      r = disk2SRs_.at(layerId).at((int)r);
    return r;
  }

  // returns bit accurate position of a stub from a given tfp region [0-8]
  GlobalPoint Setup::stubPos(bool hybrid, const FrameStub& frame, int region) const {
    GlobalPoint p;
    if (frame.first.isNull())
      return p;
    TTBV bv(frame.second);
    if (hybrid) {
      const bool barrel = this->barrel(frame.first);
      const int layerId = this->indexLayerId(frame.first);
      const GlobalPoint gp = this->stubPos(frame.first);
      const bool side = gp.z() > 0.;
      const SensorModule::Type type = this->type(frame.first);
      const int widthBend = hybridWidthsBend_.at(type);
      const int widthAlpha = hybridWidthsAlpha_.at(type);
      const int widthPhi = hybridWidthsPhi_.at(type);
      const int widthZ = hybridWidthsZ_.at(type);
      const int widthR = hybridWidthsR_.at(type);
      const double basePhi = hybridBasesPhi_.at(type);
      const double baseZ = hybridBasesZ_.at(type);
      const double baseR = hybridBasesR_.at(type);
      // parse bit vector
      bv >>= 1 + hybridWidthLayerId_ + widthBend + widthAlpha;
      double phi = bv.val(basePhi, widthPhi) - hybridRangePhi_ / 2.;
      bv >>= widthPhi;
      double z = bv.val(baseZ, widthZ, 0, true);
      bv >>= widthZ;
      double r = bv.val(baseR, widthR, 0, barrel);
      if (barrel)
        r += hybridLayerRs_.at(layerId);
      else
        z += hybridDiskZs_.at(layerId) * (side ? 1. : -1.);
      phi = deltaPhi(phi + region * baseRegion_);
      if (type == SensorModule::Disk2S) {
        r = bv.val(widthR);
        r = disk2SRs_.at(layerId).at((int)r);
      }
      p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
    } else {
      bv >>= 2 * tmttWidthInv2R_ + 2 * tmttWidthSectorEta_ + numSectorsPhi_ + tmttWidthLayer_;
      double z = (bv.val(tmttWidthZ_, 0, true) + .5) * tmttBaseZ_;
      bv >>= tmttWidthZ_;
      double phi = (bv.val(tmttWidthPhi_, 0, true) + .5) * tmttBasePhi_;
      bv >>= tmttWidthPhi_;
      double r = (bv.val(tmttWidthR_, 0, true) + .5) * tmttBaseR_;
      bv >>= tmttWidthR_;
      r = r + chosenRofPhi_;
      phi = deltaPhi(phi + region * baseRegion_);
      p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
    }
    return p;
  }

  // returns global TTStub position
  GlobalPoint Setup::stubPos(const TTStubRef& ttStubRef) const {
    const DetId detId = ttStubRef->getDetId() + offsetDetIdDSV_;
    const GeomDetUnit* det = trackerGeometry_->idToDetUnit(detId);
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
    return plane.toGlobal(topol->localPosition(mp));
  }

  // range check of dtc id
  void Setup::checkDTCId(int dtcId) const {
    if (dtcId < 0 || dtcId >= numDTCsPerRegion_ * numRegions_) {
      cms::Exception exception("out_of_range");
      exception.addContext("tt::Setup::checkDTCId");
      exception << "Used DTC Id (" << dtcId << ") "
                << "is out of range 0 to " << numDTCsPerRegion_ * numRegions_ - 1 << ".";
      throw exception;
    }
  }

  // range check of tklayout id
  void Setup::checkTKLayoutId(int tkLayoutId) const {
    if (tkLayoutId <= 0 || tkLayoutId > numDTCsPerRegion_ * numRegions_) {
      cms::Exception exception("out_of_range");
      exception.addContext("tt::Setup::checkTKLayoutId");
      exception << "Used TKLayout Id (" << tkLayoutId << ") "
                << "is out of range 1 to " << numDTCsPerRegion_ * numRegions_ << ".";
      throw exception;
    }
  }

  // range check of tfp identifier
  void Setup::checkTFPIdentifier(int tfpRegion, int tfpChannel) const {
    const bool oorRegion = tfpRegion >= numRegions_ || tfpRegion < 0;
    const bool oorChannel = tfpChannel >= numDTCsPerTFP_ || tfpChannel < 0;
    if (oorRegion || oorChannel) {
      cms::Exception exception("out_of_range");
      exception.addContext("tt::Setup::checkTFPIdentifier");
      if (oorRegion)
        exception << "Requested Processing Region "
                  << "(" << tfpRegion << ") "
                  << "is out of range 0 to " << numRegions_ - 1 << ".";
      if (oorChannel)
        exception << "Requested TFP Channel "
                  << "(" << tfpChannel << ") "
                  << "is out of range 0 to " << numDTCsPerTFP_ - 1 << ".";
      throw exception;
    }
  }
}  // namespace tt
