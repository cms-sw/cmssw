#include <initializer_list>

#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/ONNXRuntime/interface/Backend.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"

namespace cms::Ort {

  void ONNXInterface::throwUnavailableBackends(std::initializer_list<Backend> backends) {
    edm::Exception ex(edm::errors::UnavailableAccelerator);
    ex << "None of the requested backends is available in this job:";
    for (Backend backend : backends) {
      ex << ' ' << backendName(backend);
    }
    ex.addContext("Calling cms::Ort::ONNXInterface::chooseBackend()");
    throw ex;
  }

}  // namespace cms::Ort
