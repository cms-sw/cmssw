#include "FWCore/ServiceRegistry/interface/ModuleConsumesESInfo.h"

namespace edm {

  ModuleConsumesESInfo::ModuleConsumesESInfo()
      : transitionOfConsumer_(Transition::Event),
        produceMethodIDOfProducer_(0),
        isSource_(false),
        isLooper_(false),
        moduleLabelMismatch_(false) {}

  std::string_view ModuleConsumesESInfo::moduleBaseType() const {
    if (isLooper_) {
      return "ESProducerLooper";
    } else if (isSource_) {
      return "ESSource";
    }
    return "ESProducer";
  }

}  // namespace edm
