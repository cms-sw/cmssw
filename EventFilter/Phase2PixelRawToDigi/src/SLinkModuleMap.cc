// Helper function to distribute detIds to s-links and vice versa
#include "EventFilter/Phase2PixelRawToDigi/interface/SLinkModuleMap.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"

#include <algorithm>

using Phase2DAQFormatSpecification::SLINKS_PER_DTC;

SLinkModuleMap::SLinkModuleMap(const TrackerDetToDTCELinkCablingMap& cablingMap) {
  auto knownDTCIdsWithIndex = cablingMap.getKnownDTCIdsWithIndex();
  for (const auto& pair : knownDTCIdsWithIndex) {
    unsigned int dtcIndex = pair.first;
    unsigned int dtcId = pair.second;

    auto detIds = cablingMap.getAllDetIdsForDTCId(dtcId);
    std::sort(detIds.begin(), detIds.end());

    // Round-robin distribution of a DTC's modules across its S-Links.
    // Module i (in sorted detId order) goes to S-Link (i % SLINKS_PER_DTC).
    int total = static_cast<int>(detIds.size());

    // Pre-reserve per FED vectors.
    int base = total / SLINKS_PER_DTC;
    int remainder = total % SLINKS_PER_DTC;
    for (int slinkId = 0; slinkId < SLINKS_PER_DTC; ++slinkId) {
      int fedId = static_cast<int>(dtcIndex) * SLINKS_PER_DTC + slinkId;
      fedIdToDetIds_[fedId].reserve(base + ((slinkId < remainder) ? 1 : 0));
    }

    for (int i = 0; i < total; ++i) {
      int slinkId = i % SLINKS_PER_DTC;
      int fedId = static_cast<int>(dtcIndex) * SLINKS_PER_DTC + slinkId;
      uint32_t detId = detIds[i];
      fedIdToDetIds_[fedId].push_back(detId);
      detIdToFedId_[detId] = fedId;
    }
  }
}

const std::vector<uint32_t>& SLinkModuleMap::detIdsForFedId(int fedId) const {
  static const std::vector<uint32_t> kEmpty;
  auto it = fedIdToDetIds_.find(fedId);
  return (it == fedIdToDetIds_.end()) ? kEmpty : it->second;
}

int SLinkModuleMap::fedIdForDetId(uint32_t detId) const {
  auto it = detIdToFedId_.find(detId);
  return (it == detIdToFedId_.end()) ? -1 : it->second;
}
