#ifndef EventFilter_Phase2PixelRawToDigi_SLinkModuleMap_h
#define EventFilter_Phase2PixelRawToDigi_SLinkModuleMap_h

// SLinks are the DTC -> FED -> DAQ data path. Each DTC owns SLINKS_PER_DTC=16
// FED ID : fedId = dtcIndex * SLINKS_PER_DTC + slinkIdInDtc
// FIXME Later these should be defined through cablingmap DB file

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

class TrackerDetToDTCELinkCablingMap;

class SLinkModuleMap {
public:
  explicit SLinkModuleMap(const TrackerDetToDTCELinkCablingMap& cablingMap);

  const std::vector<uint32_t>& detIdsForFedId(int fedId) const;
  int fedIdForDetId(uint32_t detId) const;
  const std::map<int, std::vector<uint32_t>>& fedIdToDetIds() const { return fedIdToDetIds_; }

private:
  std::map<int, std::vector<uint32_t>> fedIdToDetIds_;
  std::unordered_map<uint32_t, int> detIdToFedId_;
};

#endif  // EventFilter_Phase2PixelRawToDigi_SLinkModuleMap_h
