#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAMultiView.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include <chrono>
#include <iomanip>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

GENERATE_SOA_LAYOUT(SoATemplate, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z))

constexpr int numViews = 2;

using SoA = SoATemplate<>;
using View = SoA::View;
using ConstView = SoA::ConstViewTemplate<cms::soa::RestrictQualify::enabled, cms::soa::RangeChecking::enabled>;
using MultiView = SoAMultiView<ConstView, numViews>;

struct kernelSingleView {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, ConstView const view, float* result, const int size) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      for (int j = 0; j < 1000; ++j) {
        auto const& element = view[i];
        result[i] += element.x() * element.y() + element.z();
      }
    }
  }
};

struct kernelMultiView {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, MultiView const view, float* result, const int size) const {
    for (auto i : cms::alpakatools::uniform_elements(acc, size)) {
      for (int j = 0; j < 1000; ++j) {
        auto const& element = view[i];
        result[i] += element.x() * element.y() + element.z();
      }
    }
  }
};

template <typename F>
double benchmark(F&& f, Queue& queue, const int iterations = 1000) {
  // Warm-up
  for (int i = 0; i < 10; ++i) {
    f();
  }

  alpaka::wait(queue);

  auto start = std::chrono::high_resolution_clock::now();
  f();
  alpaka::wait(queue);
  auto stop = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
}

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available for the " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << " backend, skipping.\n";
    return EXIT_FAILURE;
  }

  auto const& device = devices[0];

  std::cout << "Running on " << alpaka::getName(device) << std::endl;
  Queue queue(device);

  for (int potenz = 5; potenz < 26; ++potenz) {
    const int totalElements = 1 << potenz;
    const int elementsPerView = totalElements / numViews;

    PortableHostCollection<SoA> hostCollection(queue, totalElements);
    auto h_view = hostCollection.view();

    auto result_h = cms::alpakatools::make_host_buffer<float[]>(queue, totalElements);
    std::memset(result_h.data(), 0x00, totalElements * sizeof(float));

    for (int i = 0; i < totalElements; i++) {
      h_view[i].x() = static_cast<float>(i);
      h_view[i].y() = static_cast<float>(i) * 2.0f;
      h_view[i].z() = static_cast<float>(i) * 3.0f;
    }

    PortableCollection<Device, SoA> deviceCollection(queue, totalElements);
    ConstView d_Constview = deviceCollection.const_view();
    alpaka::memcpy(queue, deviceCollection.buffer(), hostCollection.buffer());

    auto result_d = cms::alpakatools::make_device_buffer<float[]>(queue, totalElements);
    alpaka::memcpy(queue, result_d, result_h);

    // Work division
    const std::size_t blockSize = 256;
    const std::size_t numberOfBlocks = cms::alpakatools::divide_up_by(totalElements, blockSize);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);

    std::vector<PortableHostCollection<SoA>> hostCollections;
    for (int i = 0; i < numViews; ++i) {
      hostCollections.emplace_back(queue, elementsPerView);
    }

    for (int i = 0; i < numViews; ++i) {
      for (int j = 0; j < elementsPerView; ++j) {
        int index = i * elementsPerView + j;
        hostCollections[i].view()[j].x() = static_cast<float>(index);
        hostCollections[i].view()[j].y() = static_cast<float>(index) * 2.0f;
        hostCollections[i].view()[j].z() = static_cast<float>(index) * 3.0f;
      }
    }

    std::vector<PortableCollection<Device, SoA>> deviceCollections;
    for (int i = 0; i < numViews; ++i) {
      deviceCollections.emplace_back(queue, elementsPerView);
    }

    MultiView multiView_h(
        hostCollections, [](const PortableHostCollection<SoA>& collection) -> auto { return collection.const_view(); });

    std::cout << "Single View size: " << d_Constview.metadata().size() << std::endl;
    std::cout << "MultiView size: " << multiView_h.size() << std::endl;

    MultiView multiView_d(deviceCollections, [](const PortableCollection<Device, SoA>& collection) -> auto {
      return collection.const_view();
    });

    for (int i = 0; i < numViews; ++i) {
      alpaka::memcpy(queue, deviceCollections[i].buffer(), hostCollections[i].buffer());
    }

    alpaka::wait(queue);

    // Run benchmarks

    double singleTime = benchmark(
        [&]() { alpaka::exec<Acc1D>(queue, workDiv, kernelSingleView{}, d_Constview, result_d.data(), totalElements); },
        queue);

    alpaka::memcpy(queue, result_h, result_d);
    alpaka::wait(queue);

    std::cout << "10th element of result_h: " << result_h[10] << std::endl;

    std::memset(result_h.data(), 0x00, totalElements * sizeof(float));
    alpaka::memcpy(queue, result_d, result_h);

    double multiTime = benchmark(
        [&]() { alpaka::exec<Acc1D>(queue, workDiv, kernelMultiView{}, multiView_d, result_d.data(), totalElements); },
        queue);

    alpaka::memcpy(queue, result_h, result_d);
    alpaka::wait(queue);

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\nBenchmark results\n";
    std::cout << "-----------------------------\n";
    std::cout << "Elements          : " << totalElements << '\n';
    std::cout << "Views             : " << numViews << '\n';
    std::cout << "Elements / view   : " << elementsPerView << '\n';
    std::cout << "Single View       : " << singleTime << " ms\n";
    std::cout << "Multi View        : " << multiTime << " ms\n";
    std::cout << "Slowdown          : " << multiTime / singleTime << "x\n";
  }

  return EXIT_SUCCESS;
}
