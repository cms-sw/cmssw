#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "CalibTracker/SiStripESProducers/interface/SiStripQualityHelpers.h"

using dqm::harvesting::DQMStore;
using dqm::harvesting::MonitorElement;

namespace {

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

  std::vector<std::pair<uint16_t, uint16_t>> getFedBadChannelList(DQMStore::IGetter& dqmStore,
                                                                  const MonitorElement* me,
                                                                  float cutoff) {
    std::vector<std::pair<uint16_t, uint16_t>> ret;
    if (me->kind() == MonitorElement::Kind::TH2F) {
      TH2F* th2 = me->getTH2F();
      float entries = getProcessedEvents(dqmStore);
      if (!entries)
        entries = th2->GetBinContent(th2->GetMaximumBin());
      for (uint16_t i = 1; i < th2->GetNbinsY() + 1; ++i) {
        for (uint16_t j = 1; j < th2->GetNbinsX() + 1; ++j) {
          if (th2->GetBinContent(j, i) > cutoff * entries) {
            edm::LogInfo("SiStripBadModuleFedErrService")
                << " [SiStripBadModuleFedErrService::getFedBadChannelList] :: FedId & Channel "
                << th2->GetYaxis()->GetBinLowEdge(i) << "  " << th2->GetXaxis()->GetBinLowEdge(j);
            ret.push_back(
                std::pair<uint16_t, uint16_t>(th2->GetYaxis()->GetBinLowEdge(i), th2->GetXaxis()->GetBinLowEdge(j)));
          }
        }
      }
    }
    return ret;
  }

}  // namespace

std::unique_ptr<SiStripQuality> sistrip::badStripFromFedErr(DQMStore::IGetter& dqmStore,
                                                            const SiStripFedCabling& fedCabling,
                                                            float cutoff) {
  auto quality = std::make_unique<SiStripQuality>();
  dqmStore.cd();
  const std::string dname{"SiStrip/ReadoutView"};
  const std::string hpath{dname + "/FedIdVsApvId"};
  if (dqmStore.dirExists(dname)) {
    MonitorElement* me = dqmStore.get(hpath);
    if (me) {
      std::map<uint32_t, std::set<int>> detectorMap;
      for (const auto& elm : getFedBadChannelList(dqmStore, me, cutoff)) {
        const uint16_t fId = elm.first;
        const uint16_t fChan = elm.second / 2;
        if ((fId == 9999) && (fChan == 9999))
          continue;

        FedChannelConnection channel = fedCabling.fedConnection(fId, fChan);
        detectorMap[channel.detId()].insert(channel.apvPairNumber());
      }

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
      quality->cleanUp();
    }
  }
  return quality;
}
