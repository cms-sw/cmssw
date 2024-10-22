#include "CondFormats/RunInfo/src/headers.h"

namespace CondFormats_RunInfo {
  struct dictionary {
    std::vector<runinfo_test::RunNumber::Item>::iterator tmp0;
    std::vector<L1TriggerScaler::Lumi>::iterator tmp1;
    //  std::vector<RunSummary::Summary>::iterator tmp2
    std::bitset<3565> bits;
    MixingModuleConfig mmc;
    MixingInputConfig mic;
  };
}  // namespace CondFormats_RunInfo
