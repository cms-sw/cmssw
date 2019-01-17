#ifndef DETECTOR_DESCRIPTION_MUON_SUB_DETECTOR_H
#define DETECTOR_DESCRIPTION_MUON_SUB_DETECTOR_H

#include <string>
#include <map>

namespace cms {
  struct MuonSubDetector {
    
    enum class SubDetector {
      barrel = 1, endcap = 2,
      rpc = 3, gem = 4, me0 = 5, nodef };
      
    const std::map<SubDetector, std::string> subDetMap {
      { barrel, "MuonDTHits" },
      { endcap, "MuonCSCHits"},
      { rpc,"MuonRPCHits" },
      { gem, "MuonGEMHits" },
      { me0, "MuonME0Hits" },
      { nodef, "" }};
  };
}

#endif
