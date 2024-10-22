#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

class HLTFiltersDQMonitor : public DQMEDAnalyzer {
public:
  explicit HLTFiltersDQMonitor(edm::ParameterSet const&);
  ~HLTFiltersDQMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  bool skipStreamByName(std::string const& streamName) const;
  bool skipPathMonitorElement(std::string const& pathName) const;
  bool skipModuleByEDMType(std::string const& moduleEDMType) const;
  bool skipModuleByType(std::string const& moduleType) const;

  std::string const folderName_;
  std::string const efficPlotNamePrefix_;
  std::string processName_;
  bool initFailed_;
  bool skipRun_;

  MonitorElement* meMenu_;
  std::unordered_map<std::string, MonitorElement*> meDatasetMap_;
  std::unordered_map<std::string, MonitorElement*> mePathMap_;

  // map of bin-label-keyword -> bin-index in MonitorElement
  std::unordered_map<std::string, size_t> binIndexMap_;

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
  edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerEventWithRefsToken_;

  HLTConfigProvider hltConfigProvider_;
};

HLTFiltersDQMonitor::HLTFiltersDQMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("folderName")),
      efficPlotNamePrefix_(iConfig.getParameter<std::string>("efficPlotNamePrefix")),
      processName_(""),
      initFailed_(false),
      skipRun_(false),
      meMenu_(nullptr) {
  auto const& triggerResultsInputTag = iConfig.getParameter<edm::InputTag>("triggerResults");

  if (triggerResultsInputTag.process().empty()) {
    edm::LogError("HLTFiltersDQMonitor") << "process not specified in HLT TriggerResults InputTag \""
                                         << triggerResultsInputTag.encode()
                                         << "\" -> plugin will not produce DQM outputs";
    initFailed_ = true;
    return;
  } else {
    processName_ = triggerResultsInputTag.process();

    triggerResultsToken_ = consumes<edm::TriggerResults>(triggerResultsInputTag);

    auto triggerEventInputTag = iConfig.getParameter<edm::InputTag>("triggerEvent");
    if (triggerEventInputTag.process().empty()) {
      triggerEventInputTag = edm::InputTag(triggerEventInputTag.label(), triggerEventInputTag.instance(), processName_);
    } else if (triggerEventInputTag.process() != processName_) {
      edm::LogWarning("HLTFiltersDQMonitor")
          << "edm::TriggerResults process name '" << processName_
          << "' differs from trigger::TriggerEvent process name '" << triggerEventInputTag.process()
          << "' -> plugin will not produce DQM outputs";
      initFailed_ = true;
      return;
    }
    triggerEventToken_ = consumes<trigger::TriggerEvent>(triggerEventInputTag);

    auto triggerEventWithRefsInputTag = iConfig.getParameter<edm::InputTag>("triggerEventWithRefs");
    if (triggerEventWithRefsInputTag.process().empty()) {
      triggerEventWithRefsInputTag =
          edm::InputTag(triggerEventWithRefsInputTag.label(), triggerEventWithRefsInputTag.instance(), processName_);
    } else if (triggerEventWithRefsInputTag.process() != processName_) {
      edm::LogWarning("HLTFiltersDQMonitor")
          << "edm::TriggerResults process name '" << processName_
          << "' differs from trigger::TriggerEventWithRefs process name '" << triggerEventWithRefsInputTag.process()
          << "' -> plugin will not produce DQM outputs";
      initFailed_ = true;
      return;
    }
    triggerEventWithRefsToken_ = mayConsume<trigger::TriggerEventWithRefs>(triggerEventWithRefsInputTag);
  }
}

void HLTFiltersDQMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  if (initFailed_) {
    return;
  }

  LogTrace("HLTFiltersDQMonitor")
      << "[HLTFiltersDQMonitor] "
      << "----------------------------------------------------------------------------------------------------";
  LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::dqmBeginRun] Run = " << iRun.id();

  // reset data members holding information from the previous run
  skipRun_ = false;

  bool hltChanged = true;
  if (hltConfigProvider_.init(iRun, iSetup, processName_, hltChanged)) {
    LogTrace("HLTFiltersDQMonitor")
        << "[HLTFiltersDQMonitor::dqmBeginRun] HLTConfigProvider initialized [processName() = "
        << hltConfigProvider_.processName() << ", tableName() = " << hltConfigProvider_.tableName()
        << ", size() = " << hltConfigProvider_.size() << "]";
  } else {
    edm::LogError("HLTFiltersDQMonitor") << "initialization of HLTConfigProvider failed for Run=" << iRun.id()
                                         << " (process=\"" << processName_
                                         << "\") -> plugin will not produce DQM outputs for this run";
    skipRun_ = true;
    return;
  }
}

void HLTFiltersDQMonitor::bookHistograms(DQMStore::IBooker& iBooker,
                                         edm::Run const& iRun,
                                         edm::EventSetup const& iSetup) {
  if (skipRun_ or initFailed_) {
    return;
  }

  // clear map of bin-label-keyword -> bin-index in MonitorElement
  binIndexMap_.clear();

  iBooker.setCurrentFolder(folderName_);

  iBooker.bookString("HLTMenu", hltConfigProvider_.tableName().c_str());

  auto hltMenuName = hltConfigProvider_.tableName();
  std::replace(hltMenuName.begin(), hltMenuName.end(), '/', '_');
  std::replace(hltMenuName.begin(), hltMenuName.end(), '.', 'p');
  while (hltMenuName.front() == '_') {
    hltMenuName.erase(0, 1);
  }

  auto const& triggerNames = hltConfigProvider_.triggerNames();

  meMenu_ = iBooker.bookProfile(efficPlotNamePrefix_ + hltMenuName,
                                "Path Efficiency",
                                triggerNames.size(),
                                0.,
                                triggerNames.size(),
                                -0.1,
                                1.1,
                                "",
                                [&triggerNames](TProfile* tprof) {
                                  for (size_t idx = 0; idx < triggerNames.size(); ++idx) {
                                    tprof->GetXaxis()->SetBinLabel(idx + 1, triggerNames[idx].c_str());
                                  }
                                });

  for (size_t idx = 0; idx < triggerNames.size(); ++idx) {
    binIndexMap_[triggerNames[idx]] = idx + 1;
  }

  LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::bookHistograms] HLTConfigProvider::size() = "
                                  << hltConfigProvider_.size()
                                  << ", HLTConfigProvider::triggerNames().size() = " << triggerNames.size();

  for (auto const& istream : hltConfigProvider_.streamNames()) {
    LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::bookHistograms] Stream = \"" << istream << "\"";

    if (this->skipStreamByName(istream)) {
      continue;
    }

    auto const& dsets = hltConfigProvider_.streamContent(istream);
    for (auto const& idset : dsets) {
      iBooker.setCurrentFolder(folderName_ + "/" + idset);
      LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::bookHistograms]   Dataset = \"" << idset << "\"";
      auto const& dsetPathNames = hltConfigProvider_.datasetContent(idset);
      auto const meDatasetName = efficPlotNamePrefix_ + idset;
      meDatasetMap_[meDatasetName] = iBooker.bookProfile(
          meDatasetName.c_str(),
          meDatasetName.c_str(),
          dsetPathNames.size(),
          0.,
          dsetPathNames.size(),
          -0.1,
          1.1,
          "",
          [&dsetPathNames, &triggerNames](TProfile* tprof) {
            for (size_t idxPath = 0; idxPath < dsetPathNames.size(); ++idxPath) {
              auto const& iPathName = dsetPathNames[idxPath];
              if (std::find(triggerNames.begin(), triggerNames.end(), iPathName) == triggerNames.end()) {
                continue;
              }
              tprof->GetXaxis()->SetBinLabel(idxPath + 1, iPathName.c_str());
            }
          });
      for (size_t idxPath = 0; idxPath < dsetPathNames.size(); ++idxPath) {
        auto const& iPathName = dsetPathNames[idxPath];
        if (std::find(triggerNames.begin(), triggerNames.end(), iPathName) == triggerNames.end()) {
          continue;
        }
        binIndexMap_[idset + "." + iPathName] = idxPath + 1;

        if (this->skipPathMonitorElement(iPathName)) {
          continue;
        }

        LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::bookHistograms]     Path = \"" << iPathName << "\"";

        auto const& moduleLabels = hltConfigProvider_.moduleLabels(iPathName);
        std::vector<std::string> mePath_binLabels;
        mePath_binLabels.reserve(moduleLabels.size());
        for (size_t iMod = 0; iMod < moduleLabels.size(); ++iMod) {
          auto const& moduleLabel = moduleLabels[iMod];

          bool skipModule = false;
          if (this->skipModuleByEDMType(hltConfigProvider_.moduleEDMType(moduleLabel)) or
              this->skipModuleByType(hltConfigProvider_.moduleType(moduleLabel))) {
            skipModule = true;
          } else if (std::find(mePath_binLabels.begin(), mePath_binLabels.end(), moduleLabel) !=
                     mePath_binLabels.end()) {
            LogDebug("HLTFiltersDQMonitor")
                << "module \"" << moduleLabel << "\" included multiple times in Path \"" << iPathName << "\""
                << "-> only 1 bin labelled \"" << moduleLabel << "\" will be created in the MonitorElement of the Path";
            skipModule = true;
          }

          if (skipModule) {
            LogTrace("HLTFiltersDQMonitor")
                << "[HLTFiltersDQMonitor::bookHistograms]       [-] Module = \"" << moduleLabel << "\"";
            continue;
          }

          mePath_binLabels.emplace_back(moduleLabel);

          LogTrace("HLTFiltersDQMonitor")
              << "[HLTFiltersDQMonitor::bookHistograms]       [bin=" << mePath_binLabels.size() << "] Module = \""
              << moduleLabel << "\"";
        }

        if (mePath_binLabels.empty()) {
          continue;
        }

        auto const mePathName = efficPlotNamePrefix_ + idset + "_" + iPathName;

        mePathMap_[mePathName] =
            iBooker.bookProfile(mePathName.c_str(),
                                iPathName.c_str(),
                                mePath_binLabels.size(),
                                0.,
                                mePath_binLabels.size(),
                                -0.1,
                                1.1,
                                "",
                                [&mePath_binLabels](TProfile* tprof) {
                                  for (size_t iMod = 0; iMod < mePath_binLabels.size(); ++iMod) {
                                    tprof->GetXaxis()->SetBinLabel(iMod + 1, mePath_binLabels[iMod].c_str());
                                  }
                                });

        for (size_t iMod = 0; iMod < mePath_binLabels.size(); ++iMod) {
          binIndexMap_[idset + "." + iPathName + "." + mePath_binLabels[iMod]] = iMod + 1;
        }
      }
    }
  }
}

void HLTFiltersDQMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (skipRun_ or initFailed_) {
    return;
  }

  LogTrace("HLTFiltersDQMonitor")
      << "[HLTFiltersDQMonitor::analyze] --------------------------------------------------------";
  LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze] Run = " << iEvent.id().run()
                                  << ", LuminosityBlock = " << iEvent.id().luminosityBlock()
                                  << ", Event = " << iEvent.id().event();

  auto const& triggerResults = iEvent.getHandle(triggerResultsToken_);

  if (not triggerResults.isValid()) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(triggerResultsToken_, labels);
    edm::LogWarning("HLTFiltersDQMonitor")
        << "invalid handle to edm::TriggerResults (InputTag: \"" << labels.module << ":" << labels.productInstance
        << ":" << labels.process << "\") -> plugin will not fill DQM outputs for this event";
    return;
  }

  // fill MonitorElement: HLT-Menu (bin: Path)
  auto const& triggerNames = hltConfigProvider_.triggerNames();
  for (auto const& iPathName : triggerNames) {
    unsigned int const pathIndex = hltConfigProvider_.triggerIndex(iPathName);
    if (pathIndex >= triggerResults->size()) {
      edm::LogError("HLTFiltersDQMonitor")
          << "[HLTFiltersDQMonitor::analyze]       "
          << "index associated to Path \"" << iPathName << "\" (" << pathIndex
          << ") is inconsistent with triggerResults::size() (" << triggerResults->size()
          << ") -> plugin will not fill bin associated to this Path in HLT-Menu MonitorElement";
      continue;
    }

    auto const foundBin = binIndexMap_.find(iPathName);
    if (foundBin == binIndexMap_.end()) {
      throw cms::Exception("HLTFiltersDQMonitorInvalidBinLabel")
          << "invalid key for bin-index map (name of Path bin in MonitorElement of HLT Menu): \"" << iPathName << "\"";
    }
    auto const ibin = foundBin->second;
    if ((0 < ibin) and (ibin <= size_t(meMenu_->getNbinsX()))) {
      auto const pathAccept = triggerResults->accept(pathIndex);
      meMenu_->Fill(ibin - 0.5, pathAccept);
    } else {
      edm::LogError("HLTFiltersDQMonitor") << "out-of-range bin index of Path \"" << iPathName
                                           << "\" in MonitorElement of HLT Menu (MonitorElement not filled): bin_key=\""
                                           << iPathName << "\", bin_index=" << ibin;
    }
  }

  auto const& triggerEventHandle = iEvent.getHandle(triggerEventToken_);
  edm::Handle<trigger::TriggerEventWithRefs> triggerEventWithRefs;

  bool useTriggerEvent = true;
  if (not triggerEventHandle.isValid()) {
    useTriggerEvent = false;

    edm::EDConsumerBase::Labels triggerEventLabels;
    labelsForToken(triggerEventToken_, triggerEventLabels);

    edm::EDConsumerBase::Labels triggerEventWithRefsLabels;
    labelsForToken(triggerEventWithRefsToken_, triggerEventWithRefsLabels);

    edm::LogInfo("HLTFiltersDQMonitor") << "invalid handle to trigger::TriggerEvent (InputTag: \""
                                        << triggerEventLabels.module << ":" << triggerEventLabels.productInstance << ":"
                                        << triggerEventLabels.process
                                        << "\"), will attempt to access trigger::TriggerEventWithRefs (InputTag:\""
                                        << triggerEventWithRefsLabels.module << ":"
                                        << triggerEventWithRefsLabels.productInstance << ":"
                                        << triggerEventWithRefsLabels.process << "\")";

    triggerEventWithRefs = iEvent.getHandle(triggerEventWithRefsToken_);
    if (not triggerEventWithRefs.isValid()) {
      edm::LogWarning("HLTFiltersDQMonitor")
          << "invalid handle to trigger::TriggerEventWithRefs (InputTag: \"" << triggerEventWithRefsLabels.module << ":"
          << triggerEventWithRefsLabels.productInstance << ":" << triggerEventWithRefsLabels.process
          << "\") -> plugin will not fill DQM outputs for this event";
      return;
    }
  }

  auto const triggerEventSize = useTriggerEvent ? triggerEventHandle->sizeFilters() : triggerEventWithRefs->size();
  LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze] useTriggerEvent = " << useTriggerEvent
                                  << ", triggerEventSize = " << triggerEventSize;

  // fill MonitorElements for PrimaryDatasets and Paths
  // loop over Streams
  for (auto const& istream : hltConfigProvider_.streamNames()) {
    LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]   Stream = \"" << istream << "\"";

    // loop over PrimaryDatasets in Stream
    auto const& dsets = hltConfigProvider_.streamContent(istream);
    for (auto const& idset : dsets) {
      LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]     Dataset = \"" << idset << "\"";

      // consider only Datasets with a MonitorElement (see bookHistograms)
      auto const meDatasetName = efficPlotNamePrefix_ + idset;
      auto const meDatasetMapFindIt = meDatasetMap_.find(meDatasetName);
      if (meDatasetMapFindIt == meDatasetMap_.end()) {
        LogDebug("HLTFiltersDQMonitor") << "No MonitorElement associated to Dataset \"" << idset << "\" in Stream \""
                                        << istream << "\" (will be ignored)";
        continue;
      }
      MonitorElement* const meDatasetProf = meDatasetMapFindIt->second;

      // loop over Paths in PrimaryDataset
      auto const& dsetPathNames = hltConfigProvider_.datasetContent(idset);
      for (auto const& iPathName : dsetPathNames) {
        unsigned int const pathIndex = hltConfigProvider_.triggerIndex(iPathName);
        if (pathIndex >= triggerResults->size()) {
          edm::LogError("HLTFiltersDQMonitor")
              << "[HLTFiltersDQMonitor::analyze]       "
              << "index associated to Path \"" << iPathName << "\" (" << pathIndex
              << ") is inconsistent with triggerResults::size() (" << triggerResults->size()
              << ") -> plugin will not fill DQM info related to this Path";
          continue;
        }
        auto const pathAccept = triggerResults->accept(pathIndex);
        LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]       "
                                        << "Path = \"" << iPathName << "\", HLTConfigProvider::triggerIndex(\""
                                        << iPathName << "\") = " << pathIndex << ", Accept = " << pathAccept;

        // fill MonitorElement: PrimaryDataset (bin: Path)
        auto const ibinKey = idset + "." + iPathName;
        auto const foundBin = binIndexMap_.find(ibinKey);
        if (foundBin == binIndexMap_.end()) {
          throw cms::Exception("HLTFiltersDQMonitorInvalidBinLabel")
              << "invalid key for bin-index map (name of Path bin in MonitorElement of Dataset): \"" << ibinKey << "\"";
        }
        auto const ibin = foundBin->second;
        if (0 < ibin and ibin <= size_t(meDatasetProf->getNbinsX())) {
          meDatasetProf->Fill(ibin - 0.5, pathAccept);
        } else {
          edm::LogError("HLTFiltersDQMonitor")
              << "out-of-range bin index of Path \"" << iPathName << "\" in MonitorElement of Dataset \"" << idset
              << "\" (MonitorElement not filled): bin_key=\"" << ibinKey << "\", bin_index=" << ibin;
        }

        // fill MonitorElement: Path (bin: filter)
        auto const mePathName = efficPlotNamePrefix_ + idset + "_" + iPathName;

        // consider only Paths with a MonitorElement
        auto const mePathMapFindIt = mePathMap_.find(mePathName);
        if (mePathMapFindIt == mePathMap_.end()) {
          LogDebug("HLTFiltersDQMonitor") << "No MonitorElement associated to Path \"" << iPathName
                                          << "\" in Dataset \"" << idset << "\" (will be ignored)";
          continue;
        }
        MonitorElement* const mePathProf = mePathMapFindIt->second;

        unsigned int indexLastFilterInPath = triggerResults->index(pathIndex) + 1;
        LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]         "
                                        << "indexLastFilterInPath = " << indexLastFilterInPath;
        // identify module corresponding to last filter executed in the Path
        while (indexLastFilterInPath > 0) {
          --indexLastFilterInPath;
          auto const& labelLastFilterInPath = hltConfigProvider_.moduleLabel(pathIndex, indexLastFilterInPath);
          auto const labelLastFilterInPathTag = edm::InputTag(labelLastFilterInPath, "", processName_);
          unsigned int const indexLastFilterInTriggerEvent =
              useTriggerEvent ? triggerEventHandle->filterIndex(labelLastFilterInPathTag)
                              : triggerEventWithRefs->filterIndex(labelLastFilterInPathTag);
          LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]           "
                                          << "indexLastFilterInPath = " << indexLastFilterInPath
                                          << ", labelLastFilterInPath = " << labelLastFilterInPath
                                          << ", indexLastFilterInTriggerEvent = " << indexLastFilterInTriggerEvent
                                          << " (triggerEventSize = " << triggerEventSize << ")";
          if (indexLastFilterInTriggerEvent < triggerEventSize) {
            if (this->skipModuleByType(hltConfigProvider_.moduleType(labelLastFilterInPath))) {
              continue;
            }
            break;
          }
        }
        // number of modules in the path
        unsigned int const nModulesInPath = hltConfigProvider_.size(pathIndex);
        LogTrace("HLTFiltersDQMonitor") << "[HLTFiltersDQMonitor::analyze]         "
                                        << "-> selected indexLastFilterInPath = " << indexLastFilterInPath
                                        << " (HLTConfigProvider::size(" << pathIndex << ") = " << nModulesInPath << ")";
        if (indexLastFilterInPath >= nModulesInPath) {
          edm::LogError("HLTFiltersDQMonitor")
              << " selected index (" << indexLastFilterInPath << ") for last filter of path \"" << iPathName
              << "\" is inconsistent with number of modules in the Path (" << nModulesInPath << ")";
          continue;
        }
        // store decision of previous filter
        bool previousFilterAccept(true);
        for (size_t modIdx = 0; modIdx < nModulesInPath; ++modIdx) {
          // each filter-bin is filled, with a 0 or 1, only when all previous filters in the Path have passed
          if (not previousFilterAccept) {
            break;
          }
          // consider only selected EDFilter modules
          auto const& moduleLabel = hltConfigProvider_.moduleLabel(pathIndex, modIdx);
          if (this->skipModuleByEDMType(hltConfigProvider_.moduleEDMType(moduleLabel)) or
              this->skipModuleByType(hltConfigProvider_.moduleType(moduleLabel))) {
            continue;
          }
          // index of module in this Path [0,nModulesInPath)
          unsigned int const slotModule = hltConfigProvider_.moduleIndex(pathIndex, moduleLabel);
          bool filterAccept = false;
          if (slotModule < indexLastFilterInPath) {
            filterAccept = true;
          } else if (slotModule == indexLastFilterInPath) {
            filterAccept = pathAccept;
          }
          LogTrace("HLTFiltersDQMonitor")
              << "[HLTFiltersDQMonitor::analyze]         "
              << "HLTConfigProvider::moduleLabel(" << pathIndex << ", " << modIdx << ") = \"" << moduleLabel
              << "\", HLTConfigProvider::moduleIndex(" << pathIndex << ", \"" << moduleLabel << "\") = " << slotModule
              << ", filterAccept = " << filterAccept << ", previousFilterAccept = " << previousFilterAccept;

          auto const ibinKey = idset + "." + iPathName + "." + moduleLabel;
          auto const foundBin = binIndexMap_.find(ibinKey);
          if (foundBin == binIndexMap_.end()) {
            throw cms::Exception("HLTFiltersDQMonitorInvalidBinLabel")
                << "invalid key for bin-index map (name of Module bin in MonitorElement of Path): \"" << ibinKey
                << "\"";
          }
          auto const ibin = foundBin->second;
          if (0 < ibin and ibin <= size_t(mePathProf->getNbinsX())) {
            mePathProf->Fill(ibin - 0.5, filterAccept);
          } else {
            edm::LogError("HLTFiltersDQMonitor")
                << "out-of-range bin index of Module \"" << moduleLabel
                << "\" in MonitorElement of Path \"iPathName\" in Dataset \"" << idset
                << "\" (MonitorElement not filled): bin_key=\"" << ibinKey << "\", bin_index=" << ibin;
          }
          previousFilterAccept = filterAccept;
        }
      }
    }
  }
}

bool HLTFiltersDQMonitor::skipStreamByName(std::string const& streamName) const {
  return ((streamName.find("Physics") == std::string::npos) and (streamName.find("Scouting") == std::string::npos) and
          (streamName.find("Parking") == std::string::npos) and (streamName != "A"));
}

bool HLTFiltersDQMonitor::skipPathMonitorElement(std::string const& pathName) const {
  return ((pathName.find("HLT_") == std::string::npos) or (pathName.find("HLT_Physics") != std::string::npos) or
          (pathName.find("HLT_Random") != std::string::npos));
}

bool HLTFiltersDQMonitor::skipModuleByEDMType(std::string const& moduleEDMType) const {
  return (moduleEDMType != "EDFilter");
}

bool HLTFiltersDQMonitor::skipModuleByType(std::string const& moduleType) const { return (moduleType == "HLTBool"); }

void HLTFiltersDQMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folderName", "HLT/Filters");
  desc.add<std::string>("efficPlotNamePrefix", "effic_");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults::HLT"));
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD::HLT"));
  desc.add<edm::InputTag>("triggerEventWithRefs", edm::InputTag("hltTriggerSummaryRAW::HLT"));
  descriptions.add("dqmHLTFiltersDQMonitor", desc);
}

DEFINE_FWK_MODULE(HLTFiltersDQMonitor);
