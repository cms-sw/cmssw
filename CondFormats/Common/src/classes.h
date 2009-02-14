#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"


namespace {
  namespace {
    struct Dummy {
      cond::DataAndSummaryWrapper<int,int> dummyI;
      cond::DataAndSummaryWrapper<double,double> dummyD;
    };

  }

}

