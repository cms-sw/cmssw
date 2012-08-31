#include "CondFormats/RunInfo/interface/RunNumber.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"
#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"
#include "CondFormats/RunInfo/interface/FillInfo.h"
namespace {
  struct dictionary {
    std::vector<runinfo_test::RunNumber::Item>::iterator tmp0;
    std::vector<L1TriggerScaler::Lumi>::iterator tmp1;
    //  std::vector<RunSummary::Summary>::iterator tmp2
    std::bitset<3565> bits;
    MixingModuleConfig mmc;
    MixingInputConfig mic;
  };
}
