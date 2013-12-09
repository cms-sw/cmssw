#include "CondFormats/Luminosity/interface/LumiSectionData.h"

namespace CondFormats_Luminosity {
  struct dictionary {
    std::vector<lumi::BunchCrossingInfo>::iterator tmp1;
    std::vector<lumi::HLTInfo>::iterator tmp2;
    std::vector<lumi::TriggerInfo>::iterator tmp3;
  };
}
