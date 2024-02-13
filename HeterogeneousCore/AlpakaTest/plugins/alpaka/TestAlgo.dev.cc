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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, portabletest::TestDeviceCollection::View view, double xvalue) const {
      // global index of the thread within the grid
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
      const portabletest::Array flags = {{6, 4, 2, 0}};

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        view.r() = 1.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, view.metadata().size())) {
        view[i] = {xvalue, 0., 0., i, flags, matrix * i};
      }
    }
  };

  class TestAlgoMultiKernel2 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestDeviceMultiCollection2::View<1> view,
                                  double xvalue) const {
      // global index of the thread within the grid
      const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

      // set this only once in the whole kernel grid
      if (thread == 0) {
        view.r2() = 2.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, view.metadata().size())) {
        view[i] = {xvalue, 0., 0., i, matrix * i};
      }
    }
  };

  class TestAlgoMultiKernel3 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestDeviceMultiCollection3::View<2> view,
                                  double xvalue) const {
      // global index of the thread within the grid
      const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

      // set this only once in the whole kernel grid
      if (thread == 0) {
        view.r3() = 3.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, view.metadata().size())) {
        view[i] = {xvalue, 0., 0., i, matrix * i};
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

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernel{}, collection.view(), xvalue);
  }

  void TestAlgo::fillMulti2(Queue& queue, portabletest::TestDeviceMultiCollection2& collection, double xvalue) const {
    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(collection->metadata().size(), items);
    uint32_t groups2 = divide_up_by(collection.view<1>().metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    auto workDiv2 = make_workdiv<Acc1D>(groups2, items);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernel{}, collection.view<portabletest::TestSoA>(), xvalue);
    alpaka::exec<Acc1D>(queue, workDiv2, TestAlgoMultiKernel2{}, collection.view<portabletest::TestSoA2>(), xvalue);
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

  void TestAlgo::fillMulti3(Queue& queue, portabletest::TestDeviceMultiCollection3& collection, double xvalue) const {
    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(collection.view<portabletest::TestSoA>().metadata().size(), items);
    uint32_t groups2 = divide_up_by(collection.view<portabletest::TestSoA2>().metadata().size(), items);
    uint32_t groups3 = divide_up_by(collection.view<portabletest::TestSoA3>().metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);
    auto workDiv2 = make_workdiv<Acc1D>(groups2, items);
    auto workDiv3 = make_workdiv<Acc1D>(groups3, items);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernel{}, collection.view<portabletest::TestSoA>(), xvalue);
    alpaka::exec<Acc1D>(queue, workDiv2, TestAlgoMultiKernel2{}, collection.view<portabletest::TestSoA2>(), xvalue);
    alpaka::exec<Acc1D>(queue, workDiv3, TestAlgoMultiKernel3{}, collection.view<portabletest::TestSoA3>(), xvalue);
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

  class TestAlgoKernelUpdateMulti2 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestSoA::ConstView input,
                                  portabletest::TestSoA2::ConstView input2,
                                  AlpakaESTestDataEDevice::ConstView esData,
                                  portabletest::TestSoA::View output,
                                  portabletest::TestSoA2::View output2) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
        output2.r2() = input2.r2();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : elements_with_stride(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        if (i < esData.size()) {
          x2 += esData.val(i) + esData.val2(i);
        }
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
    }
  };

  class TestAlgoKernelUpdateMulti3 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  portabletest::TestSoA::ConstView input,
                                  portabletest::TestSoA2::ConstView input2,
                                  portabletest::TestSoA3::ConstView input3,
                                  AlpakaESTestDataEDevice::ConstView esData,
                                  portabletest::TestSoA::View output,
                                  portabletest::TestSoA2::View output2,
                                  portabletest::TestSoA3::View output3) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
        output2.r2() = input2.r2();
        output3.r3() = input3.r3();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : elements_with_stride(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
          if (0 == i)
            printf("Setting x[0] to %f\n", x);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : elements_with_stride(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        if (i < esData.size()) {
          x2 += esData.val(i) + esData.val2(i);
        }
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
      for (int32_t i : elements_with_stride(acc, output3.metadata().size())) {
        double x3 = input3[i].x3();
        if (i < esData.size()) {
          x3 += esData.val(i) + esData.val2(i);
        }
        output3[i] = {x3, input3[i].y3(), input3[i].z3(), input3[i].id3(), input3[i].m3()};
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

  portabletest::TestDeviceMultiCollection2 TestAlgo::updateMulti2(Queue& queue,
                                                                  portabletest::TestDeviceMultiCollection2 const& input,
                                                                  AlpakaESTestDataEDevice const& esData) const {
    portabletest::TestDeviceMultiCollection2 collection{input.sizes(), queue};

    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    auto sizes = collection.sizes();
    uint32_t groups = divide_up_by(*std::max_element(sizes.begin(), sizes.end()), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        TestAlgoKernelUpdateMulti2{},
                        input.view<portabletest::TestSoA>(),
                        input.view<portabletest::TestSoA2>(),
                        esData.view(),
                        collection.view<portabletest::TestSoA>(),
                        collection.view<portabletest::TestSoA2>());

    return collection;
  }

  portabletest::TestDeviceMultiCollection3 TestAlgo::updateMulti3(Queue& queue,
                                                                  portabletest::TestDeviceMultiCollection3 const& input,
                                                                  AlpakaESTestDataEDevice const& esData) const {
    portabletest::TestDeviceMultiCollection3 collection{input.sizes(), queue};

    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    auto sizes = collection.sizes();
    uint32_t groups = divide_up_by(*std::max_element(sizes.begin(), sizes.end()), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        TestAlgoKernelUpdateMulti3{},
                        input.view<portabletest::TestSoA>(),
                        input.view<portabletest::TestSoA2>(),
                        input.view<portabletest::TestSoA3>(),
                        esData.view(),
                        collection.view<portabletest::TestSoA>(),
                        collection.view<portabletest::TestSoA2>(),
                        collection.view<portabletest::TestSoA3>());

    return collection;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
