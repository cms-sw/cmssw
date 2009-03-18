#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondFormats/Common/interface/Summary.h"

#include <vector>

namespace {
  namespace {
    struct dictionaries {
	pool::PolyPtr<std::string> d0;
    };

    struct Dummy {
      std::vector<int> vid;
      std::vector<double> vdd;
      cond::DataWrapper<std::vector<int> > dummyI;
      cond::DataWrapper<std::vector<double> >dummyD;
    };

  }

}

