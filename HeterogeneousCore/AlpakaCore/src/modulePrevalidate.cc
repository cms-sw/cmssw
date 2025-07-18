#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/modulePrevalidate.h"

namespace {
  const std::string kPSetName("alpaka");
  const char* const kComment = "PSet allows to override some Alpaka module settings per module instance.";
}  // namespace

namespace cms::alpakatools {
  void modulePrevalidate(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription descAlpaka;
    descAlpaka.addUntracked<std::string>("backend", "")
        ->setComment(
            "Set the Alpaka backend for this module instance. This parameter has an effect only when the module class "
            "name has '@alpaka' suffix, i.e. it has no effect when the Alpaka backend namespace is used explicitly. "
            "Can be empty string (for the global default), 'serial_sync', or - depending on the architecture and "
            "available hardware - 'cuda_async', 'rocm_async'");
    // Optional in order to have unset state so that python-side
    // ModuleTypeResolverAlpaka can set the global default only if the
    // per-module parameter is not set
    descAlpaka.addOptionalUntracked<bool>("synchronize")
        ->setComment(
            "On backends with an asynchronous Queue, synchronize the host and device in a blocking way at the end of "
            "acquire() and produce() functions. Has no effect on backends with synchronous Queue.");

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
