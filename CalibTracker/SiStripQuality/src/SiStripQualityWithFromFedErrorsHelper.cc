#include "TFile.h"
#include "TKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiStripQuality/interface/SiStripQualityWithFromFedErrorsHelper.h"

using dqm::harvesting::DQMStore;
using dqm::harvesting::MonitorElement;

namespace {

  void fillFedBadChannelMap(const TH2F* th2,
                            float entries,
                            float cutoff,
                            const SiStripFedCabling& fedCabling,
                            std::map<uint32_t, std::set<int>>& detectorMap) {
    for (uint16_t i = 1; i < th2->GetNbinsY() + 1; ++i) {
      for (uint16_t j = 1; j < th2->GetNbinsX() + 1; ++j) {
        if (th2->GetBinContent(j, i) > cutoff * entries) {
          const uint16_t fId = th2->GetYaxis()->GetBinLowEdge(i);
          edm::LogInfo("SiStripBadModuleFedErrService")
              << " [SiStripBadModuleFedErrService::getFedBadChannelList] :: FedId & Channel " << fId << "  "
              << th2->GetXaxis()->GetBinLowEdge(j);
          const uint16_t fChan = uint16_t(th2->GetXaxis()->GetBinLowEdge(j)) / 2;
          if (!((fId == 9999) && (fChan == 9999))) {
            const auto channel = fedCabling.fedConnection(fId, fChan);
            detectorMap[channel.detId()].insert(channel.apvPairNumber());
          }
        }
      }
    }
  }

  float getProcessedEvents(DQMStore::IGetter& dqmStore) {
    dqmStore.cd();
    const std::string dname{"SiStrip/ReadoutView"};
    const std::string hpath{dname + "/nTotalBadActiveChannels"};
    if (dqmStore.dirExists(dname)) {
      MonitorElement* me = dqmStore.get(hpath);
      if (me)
        return me->getEntries();
    }
    return 0;
  }

  std::map<uint32_t, std::set<int>> getBadChannelDetectorMap(DQMStore::IGetter& dqmStore,
                                                             const SiStripFedCabling& fedCabling,
                                                             float cutoff) {
    std::map<uint32_t, std::set<int>> detectorMap;
    dqmStore.cd();
    const std::string dname{"SiStrip/ReadoutView"};
    const std::string hpath{dname + "/FedIdVsApvId"};
    if (dqmStore.dirExists(dname)) {
      MonitorElement* me = dqmStore.get(hpath);
      if (me && (me->kind() == MonitorElement::Kind::TH2F)) {
        TH2F* th2 = me->getTH2F();
        float entries = getProcessedEvents(dqmStore);
        if (!entries) {
          edm::LogWarning("SiStripBadModuleFedErrService") << "Normalising to the largest bin";
          entries = th2->GetBinContent(th2->GetMaximumBin());
        }
        fillFedBadChannelMap(th2, entries, cutoff, fedCabling, detectorMap);
      } else {
        edm::LogError("SiStripBadModuleFedErrService") << "Could not find SiStrip/ReadoutView/FedIdVsApvId";
      }
    } else {
      edm::LogError("SiStripBadModuleFedErrService") << "Could not find SiStrip/ReadoutView";
    }
    return detectorMap;
  }

  TDirectoryFile* checkAndGetSubdir(TDirectoryFile* tdir, const std::vector<std::string>& path) {
    for (const auto& subName : path) {
      tdir = tdir->Get<TDirectoryFile>(subName.c_str());
      if (!tdir) {
        break;
      }
    }
    return tdir;
  }

  std::map<uint32_t, std::set<int>> getBadChannelDetectorMap(TFile* legacyDQMTDirFile,
                                                             unsigned int runNumber,
                                                             const SiStripFedCabling& fedCabling,
                                                             float cutoff) {
    std::map<uint32_t, std::set<int>> detectorMap;
    const auto stripROVDir = checkAndGetSubdir(
        legacyDQMTDirFile, {"DQMData", "Run " + std::to_string(runNumber), "SiStrip", "Run summary", "ReadoutView"});
    if (!stripROVDir) {
      edm::LogError("SiStripBadModuleFedErrService")
          << "Could not find SiStrip/ReadoutView directory in " << legacyDQMTDirFile->GetName() << " with run number "
          << runNumber;
    } else {
      const auto th2 = stripROVDir->Get<TH2F>("FedIdVsApvId");
      if (!th2) {
        edm::LogError("SiStripBadModuleFedErrService")
            << "Could not find SiStrip/ReadoutView/FedIdVsApvId in " << legacyDQMTDirFile->GetName();
      } else {
        float entries = 0.;
        const auto nActChan = stripROVDir->Get<TH1F>("nTotalBadActiveChannels");
        if (nActChan) {
          entries = nActChan->GetEntries();
        }
        if (!entries) {
          edm::LogWarning("SiStripBadModuleFedErrService") << "Normalising to the largest bin";
          entries = th2->GetBinContent(th2->GetMaximumBin());
        }
        fillFedBadChannelMap(th2, entries, cutoff, fedCabling, detectorMap);
      }
    }
    return detectorMap;
  }

  void fillQuality(SiStripQuality* quality, const std::map<uint32_t, std::set<int>>& detectorMap) {
    for (const auto& detElm : detectorMap) {  // pair(detId, pairs)
      SiStripQuality::InputVector theSiStripVector;
      unsigned short firstBadStrip{0};
      unsigned short fNconsecutiveBadStrips{0};
      int last_pair = -1;
      for (const auto pair : detElm.second) {
        if (last_pair == -1) {
          firstBadStrip = pair * 128 * 2;
          fNconsecutiveBadStrips = 128 * 2;
        } else if (pair - last_pair > 1) {
          theSiStripVector.push_back(quality->encode(firstBadStrip, fNconsecutiveBadStrips));
          firstBadStrip = pair * 128 * 2;
          fNconsecutiveBadStrips = 128 * 2;
        } else {
          fNconsecutiveBadStrips += 128 * 2;
        }
        last_pair = pair;
      }
      unsigned int theBadStripRange = quality->encode(firstBadStrip, fNconsecutiveBadStrips);
      theSiStripVector.push_back(theBadStripRange);

      edm::LogInfo("SiStripBadModuleFedErrService")
          << " SiStripBadModuleFedErrService::readBadComponentsFromFed "
          << " detid " << detElm.first << " firstBadStrip " << firstBadStrip << " NconsecutiveBadStrips "
          << fNconsecutiveBadStrips << " packed integer " << std::hex << theBadStripRange << std::dec;

      if (!quality->put(detElm.first, SiStripBadStrip::Range{theSiStripVector.begin(), theSiStripVector.end()})) {
        edm::LogError("SiStripBadModuleFedErrService")
            << "[SiStripBadModuleFedErrService::readBadComponentsFromFed] detid already exists";
      }
    }
  }

}  // namespace

std::unique_ptr<SiStripQuality> sistrip::badStripFromFedErr(DQMStore::IGetter& dqmStore,
                                                            const SiStripFedCabling& fedCabling,
                                                            float cutoff) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  auto quality = std::make_unique<SiStripQuality>(detInfo);
  auto detectorMap = getBadChannelDetectorMap(dqmStore, fedCabling, cutoff);
  if (!detectorMap.empty()) {
    fillQuality(quality.get(), detectorMap);
    quality->cleanUp();
  } else {
    edm::LogWarning("SiStripBadModuleFedErrService") << "Empty bad channel map from FED errors";
  }
  return quality;
}

std::unique_ptr<SiStripQuality> sistrip::badStripFromFedErrLegacyDQMFile(const std::string& fileName,
                                                                         unsigned int runNumber,
                                                                         const SiStripFedCabling& fedCabling,
                                                                         float cutoff) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  auto quality = std::make_unique<SiStripQuality>(detInfo);
  auto tdirFile = TFile::Open(fileName.c_str());
  auto detectorMap = getBadChannelDetectorMap(tdirFile, runNumber, fedCabling, cutoff);
  if (!detectorMap.empty()) {
    fillQuality(quality.get(), detectorMap);
    quality->cleanUp();
  } else {
    edm::LogWarning("SiStripBadModuleFedErrService") << "Empty bad channel map from FED errors";
  }
  return quality;
}

bool SiStripQualityWithFromFedErrorsHelper::endRun(const edm::EventSetup& iSetup) {
  if (stripQualityWatcher_.check(iSetup)) {
    if (keepCopy_) {
      mergedQuality_ = std::make_unique<SiStripQuality>(iSetup.getData(stripQualityToken_));
      if (addBadCompFromFedErr_) {
        fedCabling_ = std::make_unique<SiStripFedCabling>(iSetup.getData(fedCablingToken_));
      }
      merged_ = false;
    }
    return true;
  } else {
    return false;
  }
}

const SiStripQuality& SiStripQualityWithFromFedErrorsHelper::getMergedQuality(
    dqm::harvesting::DQMStore::IGetter& getter) {
  if (!merged_) {
    if (addBadCompFromFedErr_) {
      std::unique_ptr<SiStripQuality> fedErrQuality{};
      if (fedErrLegacyFile_.empty()) {
        edm::LogInfo("SiStripQuality") << "Adding bad components from FED errors in DQM store";
        fedErrQuality = sistrip::badStripFromFedErr(getter, *fedCabling_, fedErrCutoff_);
      } else {
        edm::LogInfo("SiStripQuality") << "Adding bad components from FED errors in legacy DQM file "
                                       << fedErrLegacyFile_;
        fedErrQuality = sistrip::badStripFromFedErrLegacyDQMFile(
            fedErrLegacyFile_, fedErrFileRunNumber_, *fedCabling_, fedErrCutoff_);
      }
      mergedQuality_->add(fedErrQuality.get());
      mergedQuality_->cleanUp();
      mergedQuality_->fillBadComponents();
    }
    merged_ = true;
  }

  return *mergedQuality_;
}
