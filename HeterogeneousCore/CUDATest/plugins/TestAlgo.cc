#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"

#include "TestAlgo.h"

namespace cudatest {

  static void testAlgoKernel(cudatest::TestHostCollection::View view, int32_t size) {
    view.r() = 1.;

    for (auto i = 0; i < size; ++i) {
      view[i] = {0., 0., 0., i};
    }
  }

  void TestAlgo::fill(cudatest::TestHostCollection& collection) const {
    testAlgoKernel(collection.view(), collection->metadata().size());
  }

}  // namespace cudatest
