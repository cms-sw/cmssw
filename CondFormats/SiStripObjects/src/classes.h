#include "CondFormats/SiStripObjects/src/headers.h"

namespace CondFormats_SiStripObjects {
  struct dictionary {
    std::vector<std::vector<FedChannelConnection> > tmp1;

#ifdef SISTRIPCABLING_USING_NEW_STRUCTURE

    //    SiStripFedCabling::Registry            temp12;

#endif

    std::vector<SiStripThreshold::Container> tmp22;
    std::vector<SiStripThreshold::DetRegistry> tmp24;

    std::vector<Phase2TrackerModule> vp2m;
  };
}  // namespace CondFormats_SiStripObjects
