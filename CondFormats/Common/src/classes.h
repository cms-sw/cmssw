#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondFormats/Common/interface/GenericSummary.h"

#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Common/interface/IOVKeysDescription.h"

#include <vector>

namespace {
  namespace {
    struct dictionaries {
	pool::PolyPtr<std::string> d0;
      pool::PolyPtr<cond::IOVProvenance> d1;
      pool::PolyPtr<cond::IOVDescription> d2;
      pool::PolyPtr<cond::IOVUserMetaData> d3;
    };

    struct Dummy {
      std::vector<int> vid;
      std::vector<float> vfd;
      std::vector<double> vdd;
      std::vector<unsigned long long> vll;
      cond::DataWrapper<std::vector<int> > dummyI;
      cond::DataWrapper<std::vector<double> >dummyD;
      cond::DataWrapper<std::vector<unsigned long long> >dummyLL;
      cond::DataWrapper<cond::BaseKeyed>dummyBK;
    };

  }

}

