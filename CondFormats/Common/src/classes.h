#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include <vector>

namespace {
  namespace {
    struct Dummy {
      std::vector<int> vid;
      std::vector<double> vdd;
      cond::DataAndSummaryWrapper<std::vector<int>,std::vector<int> > dummyI;
      cond::DataAndSummaryWrapper<std::vector<double>,std::vector<double> >dummyD;
    };

  }

}

