#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

#include "ROCmTestDeviceAdditionAlgo.h"

class ROCmTestDeviceAdditionModule : public edm::global::EDAnalyzer<> {
public:
  explicit ROCmTestDeviceAdditionModule(edm::ParameterSet const& config);
  ~ROCmTestDeviceAdditionModule() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

private:
  const uint32_t size_;
};

ROCmTestDeviceAdditionModule::ROCmTestDeviceAdditionModule(edm::ParameterSet const& config)
    : size_(config.getParameter<uint32_t>("size")) {}

void ROCmTestDeviceAdditionModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<uint32_t>("size", 1024 * 1024);
  descriptions.addWithDefaultLabel(desc);
}

void ROCmTestDeviceAdditionModule::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const {
  // require ROCm for running
  edm::Service<ROCmInterface> rocm;
  if (not rocm or not rocm->enabled()) {
    std::cout << "The ROCmService is not available or disabled, the test will be skipped.\n";
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
  for (size_t i = 0; i < size_; ++i) {
    in1_h[i] = dist(rand);
    in2_h[i] = dist(rand);
    out_h[i] = 0.;
  }

  // allocate input and output buffers on the device
  float* in1_d;
  float* in2_d;
  float* out_d;
  HIP_CHECK(hipMalloc(&in1_d, size_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&in2_d, size_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&out_d, size_ * sizeof(float)));

  // copy the input data to the device
  HIP_CHECK(hipMemcpy(in1_d, in1_h.data(), size_ * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(in2_d, in2_h.data(), size_ * sizeof(float), hipMemcpyHostToDevice));

  // fill the output buffer with zeros
  HIP_CHECK(hipMemset(out_d, 0, size_ * sizeof(float)));

  // launch the 1-dimensional kernel for vector addition
  HeterogeneousTestROCmDevicePlugins::wrapper_add_vectors_f(in1_d, in2_d, out_d, size_);

  // copy the results from the device to the host
  HIP_CHECK(hipMemcpy(out_h.data(), out_d, size_ * sizeof(float), hipMemcpyDeviceToHost));

  // wait for all the operations to complete
  HIP_CHECK(hipDeviceSynchronize());

  // check the results
  for (size_t i = 0; i < size_; ++i) {
    float sum = in1_h[i] + in2_h[i];
    assert(out_h[i] < sum + epsilon);
    assert(out_h[i] > sum - epsilon);
  }

  std::cout << "All tests passed.\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ROCmTestDeviceAdditionModule);
