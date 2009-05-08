#include "CondFormats/RunInfo/interface/RunNumber.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"
#include "CondFormats/RunInfo/interface/HLTScaler.h"
#include "CondFormats/RunInfo/interface/LuminosityInfo.h"
namespace {
  struct dictionary {
    std::vector<runinfo_test::RunNumber::Item>::iterator tmp0;
    std::vector<L1TriggerScaler::Lumi>::iterator tmp1;
    //  std::vector<RunSummary::Summary>::iterator tmp2;
    std::pair< std::string,lumi::HLTInfo > tmp2;
    std::vector< std::pair< std::basic_string<char>,lumi::HLTInfo > > tmp3;
  };
}
