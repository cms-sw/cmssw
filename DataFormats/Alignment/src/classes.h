#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include <vector>
#include <map>

namespace {
  struct dictionary {
    // not needed (not instance of template):
    // AlignmentClusterFlag                     ahf;
    edm::Wrapper<AlignmentClusterFlag>          wahf;
    AliClusterValueMap                          ahvm1;
    edm::Wrapper<AliClusterValueMap>            wahvm1;
    AliTrackTakenClusterValueMap                atthvm1;  // needed?
    edm::Wrapper<AliTrackTakenClusterValueMap>  watthvm1; // needed?
  };
}
