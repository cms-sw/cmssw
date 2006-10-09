#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/Ref.h"

namespace {
  namespace {
    std::vector<TrajectorySeed> v1;
    TrajectorySeedCollection c1;
    edm::Wrapper<TrajectorySeedCollection> w1;

edm::RefVectorIterator<std::vector<TrajectorySeed>,TrajectorySeed,edm::refhelper::FindUsingAdvance<std::vector<TrajectorySeed>,TrajectoryS
eed> > rfitr1;
    edm::Ref<TrajectorySeedCollection> s1;
    edm::RefProd<TrajectorySeedCollection> s2;
    edm::RefVector<TrajectorySeedCollection> s3;
  }
}


