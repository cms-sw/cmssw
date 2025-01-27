#include "FWCore/ServiceRegistry/interface/EventSetupConsumesInfo.h"

namespace edm {

  EventSetupConsumesInfo::EventSetupConsumesInfo()
      : transitionOfConsumer_(0),
        transitionOfProducer_(0),
        isSource_(false),
        isLooper_(false),
        moduleLabelMismatch_(false),
        mayConsumes_(false),
        mayConsumesFirstEntry_(false),
        mayConsumesNoProducts_(false) {}

  std::string_view EventSetupConsumesInfo::moduleBaseType() const {
    if (isLooper_) {
      return "ESProducerLooper";
    } else if (isSource_) {
      return "ESSource";
    }
    return "ESProducer";
  }

}  // namespace edm
