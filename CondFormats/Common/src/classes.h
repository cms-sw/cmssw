#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondFormats/Common/interface/Summary.h"

#include "CondFormats/Common/interface/IOVKeysDescription.h"

#include <vector>

namespace {
  namespace {
    struct dictionaries {
	pool::PolyPtr<std::string> d0;
    };

    struct Dummy {
      std::vector<int> vid;
      std::vector<float> vfd;
      std::vector<double> vdd;
      std::vector<unsigned long long> vll;
      cond::DataWrapper<std::vector<int> > dummyI;
      cond::DataWrapper<std::vector<double> >dummyD;
      cond::DataWrapper<std::vector<unsigned long long> >dummyLL;
    };

  }

}

