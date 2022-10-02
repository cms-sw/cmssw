#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"

#include "TestAlgo.h"

namespace cudatest {

  static void testAlgoKernel(cudatest::TestHostCollection::View view, int32_t size) {
    const cudatest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

    view.r() = 1.;

    for (auto i = 0; i < size; ++i) {
      view[i] = {0., 0., 0., i, matrix * i};
    }
  }

  void TestAlgo::fill(cudatest::TestHostCollection& collection) const {
    testAlgoKernel(collection.view(), collection->metadata().size());
  }

}  // namespace cudatest
