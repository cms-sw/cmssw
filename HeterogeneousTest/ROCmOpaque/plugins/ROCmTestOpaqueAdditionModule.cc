#include <cstddef>
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
#include "HeterogeneousCore/ROCmServices/interface/ROCmService.h"
#include "HeterogeneousTest/ROCmOpaque/interface/DeviceAdditionOpaque.h"

class ROCmTestOpaqueAdditionModule : public edm::global::EDAnalyzer<> {
public:
  explicit ROCmTestOpaqueAdditionModule(edm::ParameterSet const& config);
  ~ROCmTestOpaqueAdditionModule() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

private:
  const uint32_t size_;
};

ROCmTestOpaqueAdditionModule::ROCmTestOpaqueAdditionModule(edm::ParameterSet const& config)
    : size_(config.getParameter<uint32_t>("size")) {}

void ROCmTestOpaqueAdditionModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("size", 1024 * 1024);
  descriptions.addWithDefaultLabel(desc);
}

void ROCmTestOpaqueAdditionModule::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const {
  // require ROCm for running
  edm::Service<ROCmService> cs;
  if (not cs->enabled()) {
    std::cout << "The ROCmService is disabled, the test will be skipped.\n";
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
  for (size_t i = 0; i < size_; ++i) {
    in1[i] = dist(rand);
    in2[i] = dist(rand);
    out[i] = 0.;
  }

  // launch the 1-dimensional kernel for vector addition
  cms::rocmtest::opqaue_add_vectors_f(in1.data(), in2.data(), out.data(), size_);

  // check the results
  for (size_t i = 0; i < size_; ++i) {
    float sum = in1[i] + in2[i];
    assert(out[i] < sum + epsilon);
    assert(out[i] > sum - epsilon);
  }

  std::cout << "All tests passed.\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ROCmTestOpaqueAdditionModule);
