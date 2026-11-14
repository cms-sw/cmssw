#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace {
  struct dictionary {
    TrackerDetToDTCELinkCablingMap cabmap;
    TrackerDetToDTCELinkCablingMap::ModuleInfo moduleInfo;

    DTCELinkId dtcelinkid;

    std::unordered_map<unsigned int, DTCELinkId> unorderedMapUIntToDTC;
    std::unordered_multimap<DTCELinkId, unsigned int> unorderedMapDTCToUInt;
    std::map<unsigned int, TrackerDetToDTCELinkCablingMap::ModuleInfo> mapUIntToModuleInfo;

    std::pair<unsigned int, DTCELinkId> unorderedMapUIntToDTC_data =
        std::make_pair<unsigned int, DTCELinkId>(0, DTCELinkId());
    std::pair<DTCELinkId, unsigned int> unorderedMapDTCToUInt_data =
        std::make_pair<DTCELinkId, unsigned int>(DTCELinkId(), 0);
    std::pair<unsigned int, TrackerDetToDTCELinkCablingMap::ModuleInfo> mapUIntToModuleInfo_data =
        std::make_pair<unsigned int, TrackerDetToDTCELinkCablingMap::ModuleInfo>(
            0, TrackerDetToDTCELinkCablingMap::ModuleInfo());
  };
}  // namespace
