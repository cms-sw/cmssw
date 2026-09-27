#ifndef PhysicsTools_ONNXRuntime_test_testONNXRuntime_h
#define PhysicsTools_ONNXRuntime_test_testONNXRuntime_h

#include <string>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>

#include <onnxruntime/onnxruntime_session_options_config_keys.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

// Create the ResourceInformationService, the given accelerator service (e.g. CUDAService or ROCmService, or none) and
// the ONNXService, which uses them to find the backends and devices available in the job.
// The services are available while the returned token is used with an edm::ServiceRegistry::Operate object.
inline edm::ServiceToken makeServices(std::string const& acceleratorService = "") {
  if (not edmplugin::PluginManager::isAvailable()) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

  // the parameters of each service are validated and filled with their default values when the service is created
  std::vector<edm::ParameterSet> psets;
  for (std::string const& service :
       {std::string("ResourceInformationService"), acceleratorService, std::string("ONNXService")}) {
    if (service.empty()) {
      continue;
    }
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@service_type", service);
    psets.push_back(pset);
  }
  return edm::ServiceRegistry::createSet(psets);
}

// Check that calling the given function throws an edm::Exception with the UnavailableAccelerator category, and not
// some other exception (e.g. due to a missing service).
template <typename F>
inline void assertUnavailableAccelerator(F&& function) {
  bool thrown = false;
  try {
    function();
  } catch (edm::Exception const& e) {
    thrown = true;
    CPPUNIT_ASSERT_EQUAL(edm::errors::UnavailableAccelerator, e.categoryCode());
  }
  CPPUNIT_ASSERT(thrown);
}

// Run the test model with the given backend (and device, for Backend::cuda), for different batch sizes, and check the
// results.
inline void testModel(cms::Ort::Backend backend, int device = 0) {
  using namespace cms::Ort;

  std::string model_path = edm::FileInPath("PhysicsTools/ONNXRuntime/test/data/model.onnx").fullPath();
  auto session_options = ONNXRuntime::defaultSessionOptions(backend, device);
  if (backend != Backend::cpu) {
    // make sure that the model runs on the GPU, instead of falling back to the CPU
    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  }
  ONNXRuntime rt(model_path, &session_options);
  for (const unsigned batch_size : {1, 2, 4}) {
    FloatArrays input_values{
        std::vector<float>(batch_size * 2, 1),
    };
    FloatArrays outputs;
    CPPUNIT_ASSERT_NO_THROW(outputs = rt.run({"X"}, input_values, {}, {"Y"}, batch_size));
    CPPUNIT_ASSERT(outputs.size() == 1);
    CPPUNIT_ASSERT(outputs[0].size() == batch_size);
    for (const auto& v : outputs[0]) {
      CPPUNIT_ASSERT(v == 3);
    }
  }
}

// Test a GPU backend, using the given accelerator service (e.g. CUDAService or ROCmService) to fill the
// ResourceInformation service, and the given function (e.g. cms::cudatest::testDevices or cms::rocmtest::testDevices)
// to check if any GPU is available: if a GPU is available run the test model on the device 0, otherwise check that
// requesting the backend throws an UnavailableAccelerator exception.
// If checkUnavailableDevice is true, also check that requesting a device that is not available throws an
// UnavailableAccelerator exception.
// Note: all the checks share the same services, because the CUDAService and ROCmService reset the devices when they
// are destroyed, invalidating the state of the execution providers, that are registered only once per process.
inline void testGPUBackend(cms::Ort::Backend backend,
                           std::string const& acceleratorService,
                           bool (*testDevices)(),
                           bool checkUnavailableDevice = false) {
  using namespace cms::Ort;

  // a device index that is not expected to be available in the job
  constexpr int kUnavailableDevice = 1024;

  edm::ServiceToken token = makeServices(acceleratorService);
  edm::ServiceRegistry::Operate operate(token);
  if (testDevices()) {
    testModel(backend);
    if (checkUnavailableDevice) {
      assertUnavailableAccelerator([backend]() { ONNXRuntime::defaultSessionOptions(backend, kUnavailableDevice); });
    }
  } else {
    assertUnavailableAccelerator([backend]() { ONNXRuntime::defaultSessionOptions(backend); });
  }
}

#endif  // PhysicsTools_ONNXRuntime_test_testONNXRuntime_h
