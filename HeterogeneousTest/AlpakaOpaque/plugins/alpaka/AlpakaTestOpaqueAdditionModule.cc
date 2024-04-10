#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousTest/AlpakaOpaque/interface/alpaka/DeviceAdditionOpaque.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class AlpakaTestOpaqueAdditionModule : public edm::global::EDAnalyzer<> {
  public:
    explicit AlpakaTestOpaqueAdditionModule(edm::ParameterSet const& config);
    ~AlpakaTestOpaqueAdditionModule() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

  private:
    const uint32_t size_;
  };

  AlpakaTestOpaqueAdditionModule::AlpakaTestOpaqueAdditionModule(edm::ParameterSet const& config)
      : size_(config.getParameter<uint32_t>("size")) {}

  void AlpakaTestOpaqueAdditionModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("size", 1024 * 1024);

    // ignore the alpaka = cms.untracked.PSet(...) injected by the framework
    edm::ParameterSetDescription alpaka;
    alpaka.setAllowAnything();
    desc.addUntracked<edm::ParameterSetDescription>("alpaka", alpaka);

    descriptions.addWithDefaultLabel(desc);
  }

  void AlpakaTestOpaqueAdditionModule::analyze(edm::StreamID,
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
    std::vector<float> in1(size_);
    std::vector<float> in2(size_);
    std::vector<float> out(size_);

    // fill the input buffers with random data, and the output buffer with zeros
    for (uint32_t i = 0; i < size_; ++i) {
      in1[i] = dist(rand);
      in2[i] = dist(rand);
      out[i] = 0.;
    }

    // launch the 1-dimensional kernel for vector addition on the first available device
    test::opaque_add_vectors_f(in1.data(), in2.data(), out.data(), size_);

    // check the results
    for (uint32_t i = 0; i < size_; ++i) {
      float sum = in1[i] + in2[i];
      assert(out[i] < sum + epsilon);
      assert(out[i] > sum - epsilon);
    }

    std::cout << "All tests passed.\n";
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(AlpakaTestOpaqueAdditionModule);
