#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"
#include "HeterogeneousTest/AlpakaWrapper/interface/alpaka/DeviceAdditionWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class AlpakaTestWrapperAdditionModule : public edm::global::EDAnalyzer<> {
  public:
    explicit AlpakaTestWrapperAdditionModule(edm::ParameterSet const& config);
    ~AlpakaTestWrapperAdditionModule() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

  private:
    const uint32_t size_;
  };

  AlpakaTestWrapperAdditionModule::AlpakaTestWrapperAdditionModule(edm::ParameterSet const& config)
      : size_(config.getParameter<uint32_t>("size")) {}

  void AlpakaTestWrapperAdditionModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("size", 1024 * 1024);

    // ignore the alpaka = cms.untracked.PSet(...) injected by the framework
    edm::ParameterSetDescription alpaka;
    alpaka.setAllowAnything();
    desc.addUntracked<edm::ParameterSetDescription>("alpaka", alpaka);

    descriptions.addWithDefaultLabel(desc);
  }

  void AlpakaTestWrapperAdditionModule::analyze(edm::StreamID,
                                                edm::Event const& event,
                                                edm::EventSetup const& setup) const {
    // require a valid Alpaka backend for running
    edm::Service<ALPAKA_TYPE_ALIAS(AlpakaService)> service;
    if (not service or not service->enabled()) {
      std::cout << "The " << ALPAKA_TYPE_ALIAS_NAME(AlpakaService)
                << " is not available or disabled, the test will be skipped.\n";
      return;
    }

    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // allocate input and output host buffers
    std::vector<float> in1_h(size_);
    std::vector<float> in2_h(size_);
    std::vector<float> out_h(size_);

    // fill the input buffers with random data, and the output buffer with zeros
    for (uint32_t i = 0; i < size_; ++i) {
      in1_h[i] = dist(rand);
      in2_h[i] = dist(rand);
      out_h[i] = 0.;
    }

    // run the test on all available devices
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      Queue queue{device};

      // allocate input and output buffers on the device
      auto in1_d = cms::alpakatools::make_device_buffer<float[]>(queue, size_);
      auto in2_d = cms::alpakatools::make_device_buffer<float[]>(queue, size_);
      auto out_d = cms::alpakatools::make_device_buffer<float[]>(queue, size_);

      // copy the input data to the device
      // FIXME: pass the explicit size of type uint32_t to avoid compilation error
      // The destination view and the extent are required to have compatible index types!
      alpaka::memcpy(queue, in1_d, in1_h, size_);
      alpaka::memcpy(queue, in2_d, in2_h, size_);

      // fill the output buffer with zeros
      alpaka::memset(queue, out_d, 0);

      // launch the 1-dimensional kernel for vector addition
      test::wrapper_add_vectors_f(queue, in1_d.data(), in2_d.data(), out_d.data(), size_);

      // copy the results from the device to the host
      alpaka::memcpy(queue, out_h, out_d);

      // wait for all the operations to complete
      alpaka::wait(queue);

      // check the results
      for (uint32_t i = 0; i < size_; ++i) {
        float sum = in1_h[i] + in2_h[i];
        assert(out_h[i] < sum + epsilon);
        assert(out_h[i] > sum - epsilon);
      }
    }

    std::cout << "All tests passed.\n";
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(AlpakaTestWrapperAdditionModule);
