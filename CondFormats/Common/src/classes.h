#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"


namespace {
  namespace {
    struct Dummy {
      cond::DataAndSummaryWrapper<std::vector<int>,std::vector<int> > dummyI;
      cond::DataAndSummaryWrapper<std::vector<double>,std::vector<double> >dummyD;
    };

  }

}

