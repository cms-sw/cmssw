// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class TestAlgoKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestDeviceCollection::View view,
                                  double xvalue) const {
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
      const portabletest::Array flags = {{6, 4, 2, 0}};

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        view.r() = 1.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
        view[i] = {xvalue, 0., 0., i, flags, matrix * i};
      }
    }
  };

  class TestAlgoMultiKernel2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestDeviceMultiCollection2::View<1> view,
                                  double xvalue) const {
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        view.r2() = 2.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
        view[i] = {xvalue, 0., 0., i, matrix * i};
      }
    }
  };

  class TestAlgoMultiKernel3 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestDeviceMultiCollection3::View<2> view,
                                  double xvalue) const {
      const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};

      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        view.r3() = 3.;
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
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
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestDeviceCollection::ConstView input,
                                  AlpakaESTestDataEDevice::ConstView esData,
                                  portabletest::TestDeviceCollection::View output) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestDeviceCollection::ConstView input,
                                  TestAlgo::UpdateInfo const* updateInfo,
                                  portabletest::TestDeviceCollection::View output) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        x += updateInfo->x;
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
    }
  };

  class TestAlgoKernelUpdateMulti2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
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
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : uniform_elements(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        if (i < esData.size()) {
          x2 += esData.val(i) + esData.val2(i);
        }
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestSoA::ConstView input,
                                  portabletest::TestSoA2::ConstView input2,
                                  TestAlgo::UpdateInfo const* updateInfo,
                                  portabletest::TestSoA::View output,
                                  portabletest::TestSoA2::View output2) const {
      // set this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        output.r() = input.r();
        output2.r2() = input2.r2();
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        x += updateInfo->x;
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : uniform_elements(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        x2 += updateInfo->x;
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
    }
  };

  class TestAlgoKernelUpdateMulti3 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
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
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        if (i < esData.size()) {
          x += esData.val(i) + esData.val2(i);
          if (0 == i)
            printf("Setting x[0] to %f\n", x);
        }
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : uniform_elements(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        if (i < esData.size()) {
          x2 += esData.val(i) + esData.val2(i);
        }
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
      for (int32_t i : uniform_elements(acc, output3.metadata().size())) {
        double x3 = input3[i].x3();
        if (i < esData.size()) {
          x3 += esData.val(i) + esData.val2(i);
        }
        output3[i] = {x3, input3[i].y3(), input3[i].z3(), input3[i].id3(), input3[i].m3()};
      }
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  portabletest::TestSoA::ConstView input,
                                  portabletest::TestSoA2::ConstView input2,
                                  portabletest::TestSoA3::ConstView input3,
                                  TestAlgo::UpdateInfo const* updateInfo,
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
      for (int32_t i : uniform_elements(acc, output.metadata().size())) {
        double x = input[i].x();
        x += updateInfo->x;
        if (0 == i)
          printf("Setting x[0] to %f\n", x);
        output[i] = {x, input[i].y(), input[i].z(), input[i].id(), input[i].flags(), input[i].m()};
      }
      for (int32_t i : uniform_elements(acc, output2.metadata().size())) {
        double x2 = input2[i].x2();
        x2 += updateInfo->x;
        output2[i] = {x2, input2[i].y2(), input2[i].z2(), input2[i].id2(), input2[i].m2()};
      }
      for (int32_t i : uniform_elements(acc, output3.metadata().size())) {
        double x3 = input3[i].x3();
        x3 += updateInfo->x;
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

  portabletest::TestDeviceCollection TestAlgo::update(Queue& queue,
                                                      portabletest::TestDeviceCollection const& input,
                                                      UpdateInfo const* d_updateInfo) const {
    portabletest::TestDeviceCollection collection{input->metadata().size(), queue};

    // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
    uint32_t items = 64;

    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(collection->metadata().size(), items);

    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, TestAlgoKernelUpdate{}, input.view(), d_updateInfo, collection.view());

    return collection;
  }

  portabletest::TestDeviceMultiCollection2 TestAlgo::updateMulti2(Queue& queue,
                                                                  portabletest::TestDeviceMultiCollection2 const& input,
                                                                  UpdateInfo const* d_updateInfo) const {
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
                        d_updateInfo,
                        collection.view<portabletest::TestSoA>(),
                        collection.view<portabletest::TestSoA2>());

    return collection;
  }

  portabletest::TestDeviceMultiCollection3 TestAlgo::updateMulti3(Queue& queue,
                                                                  portabletest::TestDeviceMultiCollection3 const& input,
                                                                  UpdateInfo const* d_updateInfo) const {
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
                        d_updateInfo,
                        collection.view<portabletest::TestSoA>(),
                        collection.view<portabletest::TestSoA2>(),
                        collection.view<portabletest::TestSoA3>());

    return collection;
  }

  class TestZeroCollectionKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, portabletest::TestDeviceCollection::ConstView view) const {
      const portabletest::Matrix matrix{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
      const portabletest::Array flags = {{0, 0, 0, 0}};

      // check this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        ALPAKA_ASSERT(view.r() == 0.);
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
        auto element = view[i];
        ALPAKA_ASSERT(element.x() == 0.);
        ALPAKA_ASSERT(element.y() == 0.);
        ALPAKA_ASSERT(element.z() == 0.);
        ALPAKA_ASSERT(element.id() == 0.);
        ALPAKA_ASSERT(element.flags() == flags);
        ALPAKA_ASSERT(element.m() == matrix);
      }
    }
  };

  class TestZeroMultiCollectionKernel2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, portabletest::TestDeviceMultiCollection2::ConstView<1> view) const {
      const portabletest::Matrix matrix{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};

      // check this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        ALPAKA_ASSERT(view.r2() == 0.);
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
        auto element = view[i];
        ALPAKA_ASSERT(element.x2() == 0.);
        ALPAKA_ASSERT(element.y2() == 0.);
        ALPAKA_ASSERT(element.z2() == 0.);
        ALPAKA_ASSERT(element.id2() == 0.);
        ALPAKA_ASSERT(element.m2() == matrix);
      }
    }
  };

  class TestZeroMultiCollectionKernel3 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, portabletest::TestDeviceMultiCollection3::ConstView<2> view) const {
      const portabletest::Matrix matrix{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};

      // check this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        ALPAKA_ASSERT(view.r3() == 0.);
      }

      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : uniform_elements(acc, view.metadata().size())) {
        auto element = view[i];
        ALPAKA_ASSERT(element.x3() == 0.);
        ALPAKA_ASSERT(element.y3() == 0.);
        ALPAKA_ASSERT(element.z3() == 0.);
        ALPAKA_ASSERT(element.id3() == 0.);
        ALPAKA_ASSERT(element.m3() == matrix);
      }
    }
  };

  class TestZeroStructKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, portabletest::TestDeviceObject::Product const* data) const {
      // check this only once in the whole kernel grid
      if (once_per_grid(acc)) {
        ALPAKA_ASSERT(data->x == 0.);
        ALPAKA_ASSERT(data->y == 0.);
        ALPAKA_ASSERT(data->z == 0.);
        ALPAKA_ASSERT(data->id == 0);
      }
    }
  };

  // Check that the collection has been filled with zeroes.
  void TestAlgo::checkZero(Queue& queue, portabletest::TestDeviceCollection const& collection) const {
    // create a work division with a single block and
    //   - 32 threads with a single element per thread on a GPU backend
    //   - 32 elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(1, 32);

    // the kernel will make a strided loop over the launch grid to cover all elements in the collection
    alpaka::exec<Acc1D>(queue, workDiv, TestZeroCollectionKernel{}, collection.const_view());
  }

  // Check that the collection has been filled with zeroes.
  void TestAlgo::checkZero(Queue& queue, portabletest::TestDeviceMultiCollection2 const& collection) const {
    // create a work division with a single block and
    //   - 32 threads with a single element per thread on a GPU backend
    //   - 32 elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(1, 32);

    // the kernels will make a strided loop over the launch grid to cover all elements in the collection
    alpaka::exec<Acc1D>(queue, workDiv, TestZeroCollectionKernel{}, collection.const_view<portabletest::TestSoA>());
    alpaka::exec<Acc1D>(
        queue, workDiv, TestZeroMultiCollectionKernel2{}, collection.const_view<portabletest::TestSoA2>());
  }

  // Check that the collection has been filled with zeroes.
  void TestAlgo::checkZero(Queue& queue, portabletest::TestDeviceMultiCollection3 const& collection) const {
    // create a work division with a single block and
    //   - 32 threads with a single element per thread on a GPU backend
    //   - 32 elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(1, 32);

    // the kernels will make a strided loop over the launch grid to cover all elements in the collection
    alpaka::exec<Acc1D>(queue, workDiv, TestZeroCollectionKernel{}, collection.const_view<portabletest::TestSoA>());
    alpaka::exec<Acc1D>(
        queue, workDiv, TestZeroMultiCollectionKernel2{}, collection.const_view<portabletest::TestSoA2>());
    alpaka::exec<Acc1D>(
        queue, workDiv, TestZeroMultiCollectionKernel3{}, collection.const_view<portabletest::TestSoA3>());
  }

  // Check that the object has been filled with zeroes.
  void TestAlgo::checkZero(Queue& queue, portabletest::TestDeviceObject const& object) const {
    // create a work division with a single block and
    //   - 32 threads with a single element per thread on a GPU backend
    //   - 32 elements within a single thread on a CPU backend
    auto workDiv = make_workdiv<Acc1D>(1, 32);

    // the kernel will actually use a single thread
    alpaka::exec<Acc1D>(queue, workDiv, TestZeroStructKernel{}, object.data());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
