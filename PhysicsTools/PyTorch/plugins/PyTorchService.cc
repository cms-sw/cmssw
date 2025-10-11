#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

class PyTorchService {
public:
  PyTorchService(const edm::ParameterSet& config, edm::ActivityRegistry& registry) {
    registry.watchPreallocate(this, &PyTorchService::preallocate);
  };
  ~PyTorchService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("PyTorchService", desc);
    descriptions.setComment("Disable internal PyTorch threading model.");
  }

  void preallocate(edm::service::SystemBounds const&) {
    edm::LogInfo("PyTorchService") << "Disabling PyTorch internal threading model. "
                                      "All CPU based operations will run single-threaded.";
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
  }
};

DEFINE_FWK_SERVICE(PyTorchService);
