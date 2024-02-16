//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct vector_pushback {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, SimpleVector<int>* foo) const {
    for (auto index : uniform_elements(acc))
      foo->push_back(acc, index);
  }
};

struct vector_reset {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, SimpleVector<int>* foo) const {
    foo->reset();
  }
};

struct vector_emplace_back {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc, SimpleVector<int>* foo) const {
    for (auto index : uniform_elements(acc))
      foo->emplace_back(acc, index);
  }
};

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << ", the test will be skipped.\n";
    return 0;
  }

  // run the test on each device
  for (auto const& device : devices) {
    Queue queue(device);
    auto maxN = 10000;
    auto vec_h = make_host_buffer<cms::alpakatools::SimpleVector<int>>(queue);
    auto vec_d = make_device_buffer<cms::alpakatools::SimpleVector<int>>(queue);
    auto data_h = make_host_buffer<int[]>(queue, maxN);
    auto data_d = make_device_buffer<int[]>(queue, maxN);

    [[maybe_unused]] auto v = make_SimpleVector(maxN, data_d.data());

    // Prepare the vec object on the host
    auto tmp_vec_h = make_host_buffer<cms::alpakatools::SimpleVector<int>>(queue);
    make_SimpleVector(tmp_vec_h.data(), maxN, data_d.data());
    assert(tmp_vec_h->size() == 0);
    assert(tmp_vec_h->capacity() == static_cast<int>(maxN));

    // ... and copy the object to the device.
    alpaka::memcpy(queue, vec_d, tmp_vec_h);
    alpaka::wait(queue);

    int numBlocks = 5;
    int numThreadsPerBlock = 256;
    const auto workDiv = make_workdiv<Acc1D>(numBlocks, numThreadsPerBlock);
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, vector_pushback(), vec_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, vec_h, vec_d);
    alpaka::wait(queue);
    printf("vec_h->size()=%d, numBlocks * numThreadsPerBlock=%d, maxN=%d\n",
           vec_h->size(),
           numBlocks * numThreadsPerBlock,
           maxN);
    assert(vec_h->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, vector_reset(), vec_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, vec_h, vec_d);
    alpaka::wait(queue);

    assert(vec_h->size() == 0);

    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv, vector_emplace_back(), vec_d.data()));
    alpaka::wait(queue);

    alpaka::memcpy(queue, vec_h, vec_d);
    alpaka::wait(queue);

    assert(vec_h->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

    alpaka::memcpy(queue, data_h, data_d);
  }
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
