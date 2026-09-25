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
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

// Create the ResourceInformationService and the given accelerator service (e.g. CUDAService or ROCmService), that
// are used by cms::Ort::ONNXRuntime::defaultSessionOptions() to check which GPU backends are available in the job.
// The services are available while the returned token is used with an edm::ServiceRegistry::Operate object.
inline edm::ServiceToken makeServices(std::string const& acceleratorService) {
  if (not edmplugin::PluginManager::isAvailable()) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

  // the parameters of each service are validated and filled with their default values when the service is created
  std::vector<edm::ParameterSet> psets;
  for (std::string const& service : {std::string("ResourceInformationService"), acceleratorService}) {
    edm::ParameterSet pset;
    pset.addParameter<std::string>("@service_type", service);
    psets.push_back(pset);
  }
  return edm::ServiceRegistry::createSet(psets);
}

// Run the test model with the given backend, for different batch sizes, and check the results.
inline void testModel(cms::Ort::Backend backend) {
  using namespace cms::Ort;

  std::string model_path = edm::FileInPath("PhysicsTools/ONNXRuntime/test/data/model.onnx").fullPath();
  auto session_options = ONNXRuntime::defaultSessionOptions(backend);
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
// to check if any GPU is available: if a GPU is available run the test model, otherwise check that requesting the
// backend throws an exception.
inline void testGPUBackend(cms::Ort::Backend backend, std::string const& acceleratorService, bool (*testDevices)()) {
  using namespace cms::Ort;

  edm::ServiceToken token = makeServices(acceleratorService);
  edm::ServiceRegistry::Operate operate(token);
  if (testDevices()) {
    testModel(backend);
  } else {
    CPPUNIT_ASSERT_THROW(ONNXRuntime::defaultSessionOptions(backend), cms::Exception);
  }
}

#endif  // PhysicsTools_ONNXRuntime_test_testONNXRuntime_h
