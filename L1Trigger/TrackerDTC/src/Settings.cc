#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <set>
#include <vector>
#include <memory>

using namespace std;
using namespace edm;

namespace TrackerDTC {

  Settings::Settings(const ParameterSet& iConfig)
      :  //TrackerDTCProducer parameter sets
        paramsED_(iConfig.getParameter<ParameterSet>("ParamsED")),
        paramsRouter_(iConfig.getParameter<ParameterSet>("ParamsRouter")),
        paramsConverter_(iConfig.getParameter<ParameterSet>("ParamsConverter")),
        paramsTracker_(iConfig.getParameter<ParameterSet>("ParamsTracker")),
        paramsFW_(iConfig.getParameter<ParameterSet>("ParamsFW")),
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        // ED parameter
        inputTagTTStubDetSetVec_(paramsED_.getParameter<InputTag>("InputTagTTStubDetSetVec")),
        productBranch_(paramsED_.getParameter<string>("ProductBranch")),
        dataFormat_(paramsED_.getParameter<string>("DataFormat")),
        offsetDetIdDSV_(paramsED_.getParameter<int>("OffsetDetIdDSV")),
        offsetDetIdTP_(paramsED_.getParameter<int>("OffsetDetIdTP")),
        offsetLayerDisks_(paramsED_.getParameter<int>("OffsetLayerDisks")),
        offsetLayerId_(paramsED_.getParameter<int>("OffsetLayerId")),
        // Router parameter
        enableTruncation_(paramsRouter_.getParameter<bool>("EnableTruncation")),
        freqDTC_(paramsRouter_.getParameter<double>("FreqDTC")),
        tmpTFP_(paramsRouter_.getParameter<int>("TMP_TFP")),
        numFramesInfra_(paramsRouter_.getParameter<int>("NumFramesInfra")),
        numRoutingBlocks_(paramsRouter_.getParameter<int>("NumRoutingBlocks")),
        sizeStack_(paramsRouter_.getParameter<int>("SizeStack")),
        // Converter parameter
        widthRowLUT_(paramsConverter_.getParameter<int>("WidthRowLUT")),
        widthQoverPt_(paramsConverter_.getParameter<int>("WidthQoverPt")),
        // Tracker parameter
        numRegions_(paramsTracker_.getParameter<int>("NumRegions")),
        numOverlappingRegions_(paramsTracker_.getParameter<int>("NumOverlappingRegions")),
        numDTCsPerRegion_(paramsTracker_.getParameter<int>("NumDTCsPerRegion")),
        numModulesPerDTC_(paramsTracker_.getParameter<int>("NumModulesPerDTC")),
        tmpFE_(paramsTracker_.getParameter<int>("TMP_FE")),
        widthBend_(paramsTracker_.getParameter<int>("WidthBend")),
        widthCol_(paramsTracker_.getParameter<int>("WidthCol")),
        widthRow_(paramsTracker_.getParameter<int>("WidthRow")),
        baseBend_(paramsTracker_.getParameter<double>("BaseBend")),
        baseCol_(paramsTracker_.getParameter<double>("BaseCol")),
        baseRow_(paramsTracker_.getParameter<double>("BaseRow")),
        bendCut_(paramsTracker_.getParameter<double>("BendCut")),
        freqLHC_(paramsTracker_.getParameter<double>("FreqLHC")),
        // f/w constants
        speedOfLight_(paramsFW_.getParameter<double>("SpeedOfLight")),
        bField_(paramsFW_.getParameter<double>("BField")),
        bFieldError_(paramsFW_.getParameter<double>("BFieldError")),
        outerRadius_(paramsFW_.getParameter<double>("OuterRadius")),
        innerRadius_(paramsFW_.getParameter<double>("InnerRadius")),
        maxPitch_(paramsFW_.getParameter<double>("MaxPitch")),
        // Format specific parameter
        maxEta_(paramsFormat_.getParameter<double>("MaxEta")),
        minPt_(paramsFormat_.getParameter<double>("MinPt")),
        chosenRofPhi_(paramsFormat_.getParameter<double>("ChosenRofPhi")),
        numLayers_(paramsFormat_.getParameter<int>("NumLayers")) {
    // derived Router parameter
    numDTCs_ = numRegions_ * numDTCsPerRegion_;
    numModules_ = numRegions_ * numDTCsPerRegion_ * numModulesPerDTC_;
    numModulesPerRoutingBlock_ = numModulesPerDTC_ / numRoutingBlocks_;

    const int numFramesPerBX = freqDTC_ / freqLHC_;
    maxFramesChannelInput_ = numFramesPerBX * tmpFE_ - numFramesInfra_;
    maxFramesChannelOutput_ = numFramesPerBX * tmpTFP_ - numFramesInfra_;

    numMergedRows_ = pow(2, widthRow_ - widthRowLUT_);
    maxCot_ = sinh(maxEta_);

    rangeQoverPt_ = speedOfLight_ * bField_ / minPt_ / 1000.;

    widthLayer_ = ceil(log2(numLayers_));

    baseRegion_ = 2. * M_PI / numRegions_;

    if (dataFormat_ == "TMTT")
      tmtt_ = make_unique<SettingsTMTT>(iConfig, this);
    else if (dataFormat_ == "Hybrid")
      hybrid_ = make_unique<SettingsHybrid>(iConfig, this);
    else
      throw cms::Exception("L1TrackerDTC::Settings::Settings unknown data format requested (" + dataFormat_ + ").");

    maxQoverPt_ = (rangeQoverPt_ - baseQoverPt_) / 2.;

    const int baseDiffM = widthRowLUT_ - widthRow_;

    baseM_ = basePhi_ * pow(2., baseDiffM);
    baseC_ = basePhi_;

    widthC_ = widthPhi_;

    const double x1 = pow(2, widthRow_) * baseRow_ * maxPitch_ / 2.;
    const double x0 = x1 - pow(2, widthRowLUT_) * baseRow_ * maxPitch_;
    const double maxM = atan2(x1, innerRadius_) - atan2(x0, innerRadius_);

    widthM_ = ceil(log2(maxM / baseM_));

    // event setup
    cablingMap_ = vector<DetId>(numModules_);
    trackerGeometry_ = nullptr;
    trackerTopology_ = nullptr;
  }

  // read in detector parameter
  void Settings::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // read in MagneticField
    ESHandle<MagneticField> magneticFieldHandle;
    iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
    const MagneticField* magneticField = magneticFieldHandle.product();

    // read in TrackerGeometry
    ESHandle<TrackerGeometry> trackerGeometryHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    trackerGeometry_ = trackerGeometryHandle.product();

    // read in TrackerTopology
    ESHandle<TrackerTopology> trackerTopologyHandle;
    iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
    trackerTopology_ = trackerTopologyHandle.product();

    // read in cabling map
    ESHandle<TrackerDetToDTCELinkCablingMap> cablingMapHandle;
    iSetup.get<TrackerDetToDTCELinkCablingMapRcd>().get(cablingMapHandle);
    const TrackerDetToDTCELinkCablingMap* cablingMap = cablingMapHandle.product();

    // check magnetic field configuration
    const double bField = magneticField->inTesla(GlobalPoint(0., 0., 0.)).z();
    if (fabs(bField - bField_) > bFieldError_) {
      cms::Exception exception("Configuration",
                               "Magnetic Field from EventSetup (" + to_string(bField) + ") differs more then " +
                                   to_string(bFieldError_) + " from configured value (" + to_string(bField_) + ").");
      exception.addContext("L1TrackerDTC::Settings::beginRun");
      throw exception;
    }

    // convert TrackerDetToDTCELinkCablingMap
    enum SubDetId { pixelBarrel = 1, pixelDisks = 2 };
    TTDTC ttDTC(numRegions_, numOverlappingRegions_, numDTCsPerRegion_);
    vector<int> modIds(numDTCs_, 0);                         // module counter for each DTC
    for (const DetId& detId : trackerGeometry_->detIds()) {  // all tracker modules

      if (detId.subdetId() == pixelBarrel || detId.subdetId() == pixelDisks)  // skip pixel detector
        continue;

      if (!trackerTopology_->isLower(detId))  // skip multiple detIds per module
        continue;

      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = cablingMap->detIdToDTCELinkId(detId.rawId() + offsetDetIdTP_).first->second.dtc_id();
      const int dtcId = ttDTC.dtcId(tklId);  // track trigger dtc id [0-215]

      int& modId = modIds[dtcId];                           // DTC module id [0-71]
      const int ModId = dtcId * numModulesPerDTC_ + modId;  // track trigger module id [0-15551]

      // store extracted information
      cablingMap_[ModId] = detId;  // connection between global module id and detId

      // check configuration
      if (modId++ == numModulesPerDTC_) {
        cms::Exception exception("Configuration",
                                 "Cabling map connects more than " + to_string(numModulesPerDTC_) + " modules to DTC " +
                                     to_string(dtcId) + ".");
        exception.addContext("L1TrackerDTC::Settings::beginRun");

        throw exception;
      }
    }

    if (dataFormat_ == "Hybrid")
      hybrid_->beginRun(iRun, this);
  }

  SettingsHybrid::SettingsHybrid(const ParameterSet& iConfig, Settings* settings)
      :  // TrackerDTCFormat parameter sets
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        paramsTTStubAlgo_(iConfig.getParameter<ParameterSet>("ParamsTTStubAlgo")),
        // TTStubAlgo parameter
        productLabel_(paramsTTStubAlgo_.getParameter<string>("Label") + "@"),
        processName_(paramsTTStubAlgo_.getParameter<string>("Process")),
        baseWindowSize_(paramsTTStubAlgo_.getParameter<double>("BaseWindowSize")),
        // format specific parameter
        numRingsPS_(paramsFormat_.getParameter<vector<int> >("NumRingsPS")),
        widthsR_(paramsFormat_.getParameter<vector<int> >("WidthsR")),
        widthsZ_(paramsFormat_.getParameter<vector<int> >("WidthsZ")),
        widthsPhi_(paramsFormat_.getParameter<vector<int> >("WidthsPhi")),
        widthsAlpha_(paramsFormat_.getParameter<vector<int> >("WidthsAlpha")),
        widthsBend_(paramsFormat_.getParameter<vector<int> >("WidthsBend")),
        rangesR_(paramsFormat_.getParameter<vector<double> >("RangesR")),
        rangesZ_(paramsFormat_.getParameter<vector<double> >("RangesZ")),
        rangesAlpha_(paramsFormat_.getParameter<vector<double> >("RangesAlpha")),
        layerRs_(paramsFormat_.getParameter<vector<double> >("LayerRs")),
        diskZs_(paramsFormat_.getParameter<vector<double> >("DiskZs")),
        disk2SRsSet_(paramsFormat_.getParameter<vector<ParameterSet> >("Disk2SRsSet")) {
    disk2SRs_.reserve(disk2SRsSet_.size());
    for (const auto& pSet : disk2SRsSet_)
      disk2SRs_.emplace_back(pSet.getParameter<vector<double> >("Disk2SRs"));

    function<bool(double, double)> comp = [](const double& lhs, const double& rhs) {
      return lhs > 0. ? (rhs > 0. ? lhs < rhs : true) : false;
    };

    const double rangeRT = 2. * max(fabs(settings->outerRadius_ - settings->chosenRofPhi_),
                                    fabs(settings->innerRadius_ - settings->chosenRofPhi_));
    const double rangePhi = settings->baseRegion_ + rangeRT * settings->rangeQoverPt_ / 2.;

    basesZ_.reserve(numSensorTypes);
    for (int type = 0; type < numSensorTypes; type++)
      basesZ_.push_back(rangesZ_[type] / pow(2., widthsZ_[type]));

    basesR_.reserve(numSensorTypes);
    for (int type = 0; type < numSensorTypes; type++)
      basesR_.push_back(rangesR_[type] / pow(2., widthsR_[type]));

    basesPhi_.reserve(numSensorTypes);
    for (int type = 0; type < numSensorTypes; type++)
      basesPhi_.push_back(rangePhi / pow(2., widthsPhi_[type]));

    basesAlpha_.reserve(numSensorTypes);
    for (int type = 0; type < numSensorTypes; type++)
      basesAlpha_.push_back(rangesAlpha_[type] / pow(2., widthsAlpha_[type]));

    numsUnusedBits_.reserve(numSensorTypes);
    for (int type = 0; type < numSensorTypes; type++)
      numsUnusedBits_.push_back(TTBV::S - widthsR_[type] - widthsZ_[type] - widthsPhi_[type] - widthsAlpha_[type] -
                                widthsBend_[type] - settings->widthLayer_ - 1);

    layerIdEncodings_.reserve(settings->numDTCs_);

    int& widthR = settings->widthR_;
    int& widthZ = settings->widthZ_;
    int& widthPhi = settings->widthPhi_;
    int& widthEta = settings->widthEta_;

    double& baseZ = settings->baseZ_;
    double& baseR = settings->baseR_;
    double& basePhi = settings->basePhi_;
    double& baseQoverPt = settings->baseQoverPt_;

    widthR = *max_element(widthsR_.begin(), widthsR_.end());
    widthZ = *max_element(widthsZ_.begin(), widthsZ_.end());
    widthPhi = *max_element(widthsPhi_.begin(), widthsPhi_.end());
    widthEta = 0;

    baseZ = *min_element(basesZ_.begin(), basesZ_.end(), comp);
    baseR = *min_element(basesR_.begin(), basesR_.end(), comp);
    basePhi = *min_element(basesPhi_.begin(), basesPhi_.end(), comp);
    baseQoverPt = settings->rangeQoverPt_ / pow(2., settings->widthQoverPt_);
  }

  void SettingsHybrid::beginRun(const Run& iRun, Settings* settings) {
    // get iConfig of used TTStubAlgorithm
    const ParameterSet* ttStubAlgorithmPSet = nullptr;
    const pset::Registry* psetRegistry = pset::Registry::instance();
    for (const ProcessConfiguration& pc : iRun.processHistory()) {
      if (!processName_.empty() && processName_ != pc.processName())
        continue;

      const ParameterSet* processPset = psetRegistry->getMapped(pc.parameterSetID());
      if (processPset && processPset->exists(productLabel_))
        ttStubAlgorithmPSet = &processPset->getParameterSet(productLabel_);
    }

    if (!ttStubAlgorithmPSet) {
      cms::Exception exception("Configuration", "TTStub algo config not found in process history.");
      exception.addContext("L1TrackerDTC::Settings::beginRun");

      throw exception;
    }

    const bool performZMatchingPS = ttStubAlgorithmPSet->getParameter<bool>("zMatchingPS");
    const bool performZMatching2S = ttStubAlgorithmPSet->getParameter<bool>("zMatching2S");

    numTiltedLayerRings_ = ttStubAlgorithmPSet->getParameter<vector<double> >("NTiltedRings");
    windowSizeBarrelLayers_ = ttStubAlgorithmPSet->getParameter<vector<double> >("BarrelCut");

    const vector<ParameterSet> pSetsEncapDisks =
        ttStubAlgorithmPSet->getParameter<vector<ParameterSet> >("EndcapCutSet");
    const vector<ParameterSet> pSetsTiltedLayer =
        ttStubAlgorithmPSet->getParameter<vector<ParameterSet> >("TiltedBarrelCutSet");

    windowSizeTiltedLayerRings_.reserve(pSetsTiltedLayer.size());
    for (const auto& pSet : pSetsTiltedLayer)
      windowSizeTiltedLayerRings_.emplace_back(pSet.getParameter<vector<double> >("TiltedCut"));

    windowSizeEndcapDisksRings_.reserve(pSetsEncapDisks.size());
    for (const auto& pSet : pSetsEncapDisks)
      windowSizeEndcapDisksRings_.emplace_back(pSet.getParameter<vector<double> >("EndcapCut"));

    int maxWindowSize(0);
    for (const auto& windowss : {windowSizeTiltedLayerRings_, windowSizeEndcapDisksRings_, {windowSizeBarrelLayers_}})
      for (const auto& windows : windowss)
        for (const auto& window : windows)
          maxWindowSize = max(maxWindowSize, (int)(window / baseWindowSize_));
    bendEncodingsPS_.reserve(maxWindowSize);
    bendEncodings2S_.reserve(maxWindowSize);

    const TrackerGeometry* trackerGeometry = settings->trackerGeometry_;
    const TrackerTopology* trackerTopology = settings->trackerTopology_;

    const TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> ttStubAlgorithm(trackerGeometry,
                                                                           trackerTopology,
                                                                           windowSizeBarrelLayers_,
                                                                           windowSizeEndcapDisksRings_,
                                                                           windowSizeTiltedLayerRings_,
                                                                           numTiltedLayerRings_,
                                                                           performZMatchingPS,
                                                                           performZMatching2S);

    // create layer encodings
    layerIdEncodings_.reserve(settings->numDTCs_);
    for (int dtcId = 0; dtcId < settings->numDTCs_; dtcId++) {
      auto begin = next(settings->cablingMap_.begin(), dtcId * settings->numModulesPerDTC_);
      auto end = next(begin, settings->numModulesPerDTC_);
      auto last = find_if(begin, end, [](const DetId& detId) { return detId.null(); });
      if (last < end)
        end = last;

      // assess layerIds connected to this DTC
      set<int> layerIds;
      for (auto it = begin; it < end; it++)
        layerIds.insert(it->subdetId() == StripSubdetector::TOB
                            ? trackerTopology->layer(*it)
                            : trackerTopology->tidWheel(*it) + settings->offsetLayerDisks_);

      if ((int)layerIds.size() > settings->numLayers_) {  // check configuration

        cms::Exception exception("overflow",
                                 "Cabling map connects more than " + to_string(settings->numLayers_) +
                                     " layers to DTC " + to_string(dtcId) + ".");
        exception.addContext("Converter::Converter");

        throw exception;
      }

      // index = decoded layerId, value = encoded layerId
      layerIdEncodings_.emplace_back(layerIds.begin(), layerIds.end());
    }

    // create bend encodings
    for (const bool& ps : {false, true}) {
      vector<vector<double> >& bendEncodings = ps ? bendEncodingsPS_ : bendEncodings2S_;
      bendEncodings.reserve(maxWindowSize + 1);
      for (int window = 0; window < maxWindowSize + 1; window++) {
        set<double> bendEncoding;
        for (int bend = 0; bend < window + 1; bend++)
          bendEncoding.insert(ttStubAlgorithm.degradeBend(ps, window, bend));

        // index = encoded bend, value = decoded bend
        bendEncodings.emplace_back(bendEncoding.begin(), bendEncoding.end());
      }
    }
  }

  SettingsTMTT::SettingsTMTT(const ParameterSet& iConfig, Settings* settings)
      :  //TrackerDTCFormat parameter sets
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        // format specific parameter
        numSectorsPhi_(paramsFormat_.getParameter<int>("NumSectorsPhi")),
        numBinsQoverPt_(paramsFormat_.getParameter<int>("NumBinsQoverPt")),
        numBinsPhiT_(paramsFormat_.getParameter<int>("NumBinsPhiT")),
        chosenRofZ_(paramsFormat_.getParameter<double>("ChosenRofZ")),
        beamWindowZ_(paramsFormat_.getParameter<double>("BeamWindowZ")),
        halfLength_(paramsFormat_.getParameter<double>("HalfLength")),
        bounderiesEta_(paramsFormat_.getParameter<vector<double> >("BounderiesEta")) {
    numSectorsEta_ = bounderiesEta_.size() - 1;                    // number of eta sectors used during track finding
    maxZT_ = settings->maxCot_ * chosenRofZ_;                      // cut on zT
    baseSector_ = settings->baseRegion_ / (double)numSectorsPhi_;  // width of phi sector in rad
    widthQoverPtBin_ = ceil(log2(numBinsQoverPt_));                // number of bits used for stub q over pt

    int& widthR = settings->widthR_;
    int& widthPhi = settings->widthPhi_;
    int& widthZ = settings->widthZ_;
    int& widthEta = settings->widthEta_;

    widthR = paramsFormat_.getParameter<int>("WidthR");
    widthPhi = paramsFormat_.getParameter<int>("WidthPhi");
    widthZ = paramsFormat_.getParameter<int>("WidthZ");
    widthEta = ceil(log2(numSectorsEta_));

    numUnusedBits_ = TTBV::S - 1 - widthR - widthPhi - widthZ - 2 * widthQoverPtBin_ - 2 * widthEta - numSectorsPhi_ -
                     settings->widthLayer_;

    double& baseQoverPt = settings->baseQoverPt_;
    double& baseR = settings->baseR_;
    double& baseZ = settings->baseZ_;
    double& basePhi = settings->basePhi_;

    baseQoverPtBin_ = settings->rangeQoverPt_ / numBinsQoverPt_;

    const int baseShiftQoverPt = widthQoverPtBin_ - settings->widthQoverPt_;

    baseQoverPt = baseQoverPtBin_ * pow(2., baseShiftQoverPt);

    const double basePhiT = baseSector_ / numBinsPhiT_;

    const double baseRgen = basePhiT / baseQoverPtBin_;
    const double rangeR = settings->outerRadius_ - settings->innerRadius_;
    const int baseShiftR = ceil(log2(rangeR / baseRgen / pow(2., widthR)));

    baseR = baseRgen * pow(2., baseShiftR);

    const double rangeZ = 2. * halfLength_;
    const int baseShiftZ = ceil(log2(rangeZ / baseR / pow(2., widthZ)));

    baseZ = baseR * pow(2., baseShiftZ);

    const double rangePhi = settings->baseRegion_ + settings->rangeQoverPt_ * baseR * pow(2., widthR) / 4.;
    const int baseShiftPhi = ceil(log2(rangePhi / basePhiT / pow(2., widthPhi)));

    basePhi = basePhiT * pow(2., baseShiftPhi);
  }

}  // namespace TrackerDTC