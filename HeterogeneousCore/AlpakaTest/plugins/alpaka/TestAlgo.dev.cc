// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class TestAlgoKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestDeviceCollection::View view,
                                  int32_t size,
                                  double xvalue) const {
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
      const portabletest::Array flags = {{6, 4, 2, 0}};

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        view.r() = 1.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, size)) {
        view[i] = {xvalue, 0., 0., i, flags, matrix * i};
      }
    }
  };

  void TestAlgo::fill(Queue& queue, portabletest::TestDeviceCollection& collection, double xvalue) const {
    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(collection->metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernel{}, collection.view(), collection->metadata().size(), xvalue);
  }

  class TestAlgoStructKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestDeviceObject::Product* data,
                                  double x,
                                  double y,
                                  double z,
                                  int32_t id) const {
      // run on a single thread
      if (once_per_grid(acc)) {
        data->x = x;
        data->y = y;
        data->z = z;
        data->id = id;
      }
    }
  };

  void TestAlgo::fillObject(
      Queue& queue, portabletest::TestDeviceObject& object, double x, double y, double z, int32_t id) const {
    // run on a single thread
    auto workDiv = make_workdiv<Acc1D>(1, 1);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoStructKernel{}, object.data(), x, y, z, id);
  }

  class TestAlgoKernelUpdate {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestDeviceCollection::ConstView input,
                                  AlpakaESTestDataEDevice::ConstView esData,
                                  portabletest::TestDeviceCollection::View output) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
    }
  };

  portabletest::TestDeviceCollection TestAlgo::update(Queue& queue,
                                                      portabletest::TestDeviceCollection const& input,
                                                      AlpakaESTestDataEDevice const& esData) const {
    portabletest::TestDeviceCollection collection{input->metadata().size(), queue};

    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(collection->metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernelUpdate{}, input.view(), esData.view(), collection.view());

    return collection;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
