#include "FWCore/ServiceRegistry/interface/ESModuleConsumesInfo.h"

namespace edm {

  ESModuleConsumesInfo::ESModuleConsumesInfo()
      : produceMethodIDOfConsumer_(0),
        produceMethodIDOfProducer_(0),
        isSource_(false),
        isLooper_(false),
        moduleLabelMismatch_(false),
        mayConsumes_(false),
        mayConsumesFirstEntry_(false),
        mayConsumesNoProducts_(false) {}

  std::string_view ESModuleConsumesInfo::moduleBaseType() const {
    if (isLooper_) {
      return "ESProducerLooper";
    } else if (isSource_) {
      return "ESSource";
    }
    return "ESProducer";
  }

}  // namespace edm
