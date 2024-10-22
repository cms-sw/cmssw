#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/module_backend_config.h"

namespace {
  const std::string kPSetName("alpaka");
  const char* const kComment =
      "PSet allows to override the Alpaka backend per module instance. Has an effect only when the module class name "
      "has '@alpaka' suffix, i.e. has no effect when the Alpaka backend namespace is used explicitly.";
}  // namespace

namespace cms::alpakatools {
  void module_backend_config(edm::ConfigurationDescriptions& iDesc) {
    // the code below leads to 'alpaka = untracked.PSet(backend = untracked.string)' to be added to the generated cfi files
    // TODO: I don't know if this is a desired behavior for HLT
    edm::ParameterSetDescription descAlpaka;
    descAlpaka.addUntracked<std::string>("backend", "")
        ->setComment(
            "Alpaka backend for this module. Can be empty string (for the global default), 'serial_sync', or "
            " - depending on the architecture and available hardware - 'cuda_async', 'rocm_async'");

    if (iDesc.defaultDescription()) {
      if (iDesc.defaultDescription()->isLabelUnused(kPSetName)) {
        iDesc.defaultDescription()
            ->addUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)
            ->setComment(kComment);
      }
    }
    for (auto& v : iDesc) {
      if (v.second.isLabelUnused(kPSetName)) {
        v.second.addUntracked<edm::ParameterSetDescription>(kPSetName, descAlpaka)->setComment(kComment);
      }
    }
  }
}  // namespace cms::alpakatools
