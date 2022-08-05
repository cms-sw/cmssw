#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  void ESProducer::throwSomeNullException() {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "The Alpaka backend has multiple devices. The device-specific produce() function returned a null product "
           "for some of the devices of the backend, but not all. This is not currently supported.";
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
