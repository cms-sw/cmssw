#include "L1Trigger/TrackerDTC/interface/SettingsHybrid.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <set>
#include <vector>
#include <memory>
#include <iterator>

using namespace std;
using namespace edm;

namespace trackerDTC {

  SettingsHybrid::SettingsHybrid(const ParameterSet& iConfig, Settings* settings)
      :  // TrackerDTCFormat parameter sets
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        paramsTTStubAlgo_(iConfig.getParameter<ParameterSet>("ParamsTTStubAlgo")),
        // TTStubAlgo parameter
        checkHistory_(paramsTTStubAlgo_.getParameter<bool>("CheckHistory")),
        productLabel_(paramsTTStubAlgo_.getParameter<string>("Label") + "@"),
        processName_(paramsTTStubAlgo_.getParameter<string>("Process")),
        baseWindowSize_(paramsTTStubAlgo_.getParameter<double>("BaseWindowSize")),
        // format specific parameter
        numRingsPS_(paramsFormat_.getParameter<vector<int>>("NumRingsPS")),
        widthsR_(paramsFormat_.getParameter<vector<int>>("WidthsR")),
        widthsZ_(paramsFormat_.getParameter<vector<int>>("WidthsZ")),
        widthsPhi_(paramsFormat_.getParameter<vector<int>>("WidthsPhi")),
        widthsAlpha_(paramsFormat_.getParameter<vector<int>>("WidthsAlpha")),
        widthsBend_(paramsFormat_.getParameter<vector<int>>("WidthsBend")),
        rangesR_(paramsFormat_.getParameter<vector<double>>("RangesR")),
        rangesZ_(paramsFormat_.getParameter<vector<double>>("RangesZ")),
        rangesAlpha_(paramsFormat_.getParameter<vector<double>>("RangesAlpha")),
        layerRs_(paramsFormat_.getParameter<vector<double>>("LayerRs")),
        diskZs_(paramsFormat_.getParameter<vector<double>>("DiskZs")),
        disk2SRsSet_(paramsFormat_.getParameter<vector<ParameterSet>>("Disk2SRsSet")) {
    disk2SRs_.reserve(disk2SRsSet_.size());
    for (const auto& pSet : disk2SRsSet_)
      disk2SRs_.emplace_back(pSet.getParameter<vector<double>>("Disk2SRs"));

    auto comp = [](const double& lhs, const double& rhs) { return lhs > 0. ? (rhs > 0. ? lhs < rhs : true) : false; };

    const double rangeRT = 2. * max(abs(settings->outerRadius_ - settings->chosenRofPhi_),
                                    abs(settings->innerRadius_ - settings->chosenRofPhi_));
    const double rangePhi = settings->baseRegion_ + rangeRT * settings->rangeQoverPt_ / 2.;

    bool error(false);
    for (const vector<int>& vec : {widthsR_, widthsZ_, widthsPhi_})
      if (vec.size() != numSensorTypes)
        error = true;
    for (const vector<double>& vec : {rangesR_, rangesZ_})
      if (vec.size() != numSensorTypes)
        error = true;
    if (error) {
      cms::Exception exception("LogicError");
      exception << "Expect for each sensor type specification of width and range of stub coordinates.";
      exception.addContext("trackerDTC::SettingsHybrid::SettingsHybrid");
      throw exception;
    }

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

    settings->widthR_ = *max_element(widthsR_.begin(), widthsR_.end());
    settings->widthZ_ = *max_element(widthsZ_.begin(), widthsZ_.end());
    settings->widthPhi_ = *max_element(widthsPhi_.begin(), widthsPhi_.end());
    settings->widthEta_ = 0;

    settings->baseZ_ = *min_element(basesZ_.begin(), basesZ_.end(), comp);
    settings->baseR_ = *min_element(basesR_.begin(), basesR_.end(), comp);
    settings->basePhi_ = *min_element(basesPhi_.begin(), basesPhi_.end(), comp);
    settings->baseQoverPt_ = settings->rangeQoverPt_ / pow(2., settings->widthQoverPt_);
  }

  // check current coniguration consistency with input configuration
  void SettingsHybrid::checkConfiguration(Settings* settings) const {
    if (!checkHistory_)
      return;
    // get iConfig of used TTStubAlgorithm in input producer
    const ParameterSet* historyTTStubAlgorithmPSet = nullptr;
    const pset::Registry* psetRegistry = pset::Registry::instance();
    for (const ProcessConfiguration& pc : settings->processHistory_) {
      if (processName_ != pc.processName())
        continue;
      const ParameterSet* processPset = psetRegistry->getMapped(pc.parameterSetID());
      if (processPset && processPset->exists(productLabel_))
        historyTTStubAlgorithmPSet = &processPset->getParameterSet(productLabel_);
    }
    if (!historyTTStubAlgorithmPSet) {
      cms::Exception exception("Configuration");
      exception << "TTStub algo config not found in process history.";
      exception << "Searched for process " << processName_ << " and label " << productLabel_ << ".";
      exception.addContext("trackerDTC::SettingsHybrid::checkConfiguration");
      throw exception;
    }
    if (settings->handleTTStubAlgorithm_.description()->pid_ != historyTTStubAlgorithmPSet->id()) {
      cms::Exception exception("Configuration");
      exception << "Configured TTStubAlgorithm inconsistent with used TTStubAlgorithm during stub production.";
      exception.addContext("trackerDTC::SettingsHybrid::checkConfiguration");
      throw exception;
    }
  }

  void SettingsHybrid::createEncodingsLayer(Settings* settings) {
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
                            ? settings->trackerTopology_->layer(*it)
                            : settings->trackerTopology_->tidWheel(*it) + settings->offsetLayerDisks_);
      // check configuration
      if ((int)layerIds.size() > settings->numLayers_) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << settings->numLayers_ << " layers to DTC " << dtcId << ".";
        exception.addContext("trackerDTC::SettingsHybrid::beginRun");
        throw exception;
      }
      // index = decoded layerId, value = encoded layerId
      layerIdEncodings_.emplace_back(layerIds.begin(), layerIds.end());
    }
  }

  void SettingsHybrid::createEncodingsBend(Settings* settings) {
    // get configuration of TTStubAlgorithm
    const ParameterSet& pSet = getParameterSet(settings->handleTTStubAlgorithm_.description()->pid_);
    numTiltedLayerRings_ = pSet.getParameter<vector<double>>("NTiltedRings");
    windowSizeBarrelLayers_ = pSet.getParameter<vector<double>>("BarrelCut");
    const vector<ParameterSet>& pSetsEncapDisks = pSet.getParameter<vector<ParameterSet>>("EndcapCutSet");
    const vector<ParameterSet>& pSetsTiltedLayer = pSet.getParameter<vector<ParameterSet>>("TiltedBarrelCutSet");
    windowSizeTiltedLayerRings_.reserve(pSetsTiltedLayer.size());
    for (const auto& pSet : pSetsTiltedLayer)
      windowSizeTiltedLayerRings_.emplace_back(pSet.getParameter<vector<double>>("TiltedCut"));
    windowSizeEndcapDisksRings_.reserve(pSetsEncapDisks.size());
    for (const auto& pSet : pSetsEncapDisks)
      windowSizeEndcapDisksRings_.emplace_back(pSet.getParameter<vector<double>>("EndcapCut"));
    int maxWindowSize(0);
    for (const auto& windowss : {windowSizeTiltedLayerRings_, windowSizeEndcapDisksRings_, {windowSizeBarrelLayers_}})
      for (const auto& windows : windowss)
        for (const auto& window : windows)
          maxWindowSize = max(maxWindowSize, (int)(window / baseWindowSize_));
    // create bend encodings
    const TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>* ttStubAlgorithm =
        dynamic_cast<const TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>*>(
            settings->handleTTStubAlgorithm_.product());
    bendEncodingsPS_.reserve(maxWindowSize);
    bendEncodings2S_.reserve(maxWindowSize);
    for (const bool& ps : {false, true}) {
      vector<vector<double>>& bendEncodings = ps ? bendEncodingsPS_ : bendEncodings2S_;
      bendEncodings.reserve(maxWindowSize + 1);
      for (int window = 0; window < maxWindowSize + 1; window++) {
        set<double> bendEncoding;
        for (int bend = 0; bend < window + 1; bend++)
          bendEncoding.insert(ttStubAlgorithm->degradeBend(ps, window, bend));
        // index = encoded bend, value = decoded bend
        bendEncodings.emplace_back(bendEncoding.begin(), bendEncoding.end());
      }
    }
  }

}  // namespace trackerDTC