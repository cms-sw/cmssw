#ifndef FWCore_ServiceRegistry_ModuleConsumesESInfo_h
#define FWCore_ServiceRegistry_ModuleConsumesESInfo_h

/**\class edm::ModuleConsumesESInfo

   Description: Contains information about a product
   a module declares it will consume from the EventSetup.

   There will be one object per consumes call.

   Usage: These are typically available from the PathsAndConsumesOfModules
   object passed as an argument to a Service callback.
*/
//
// Original Author: W. David Dagenhart
//         Created: 11/14/2024

#include "FWCore/Utilities/interface/Transition.h"

#include <string_view>

namespace edm {
  struct ModuleConsumesESInfo {
    ModuleConsumesESInfo();

    std::string_view moduleBaseType() const;

    // These are created by EDConsumerBase.

    // An empty moduleLabel_ indicates there is not an
    // ESProducer, ESSource, or Looper configured to deliver
    // the product with the requested EventSetupRecordKey and
    // DataKey. Even if such a module exists, if the requested
    // module label does not match an exception will be thrown
    // if there is an attempt to actually get the data (or
    // when the ESHandle is dereferenced if one gets that).
    std::string_view eventSetupRecordType_;
    std::string_view productType_;
    std::string_view productLabel_;
    std::string_view requestedModuleLabel_;
    std::string_view moduleType_;
    std::string_view moduleLabel_;
    Transition transitionOfConsumer_;
    unsigned int produceMethodIDOfProducer_;
    bool isSource_;
    bool isLooper_;
    bool moduleLabelMismatch_;
  };
}  // namespace edm
#endif
