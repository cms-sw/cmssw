#ifndef FWCore_ServiceRegistry_ESModuleConsumesInfo_h
#define FWCore_ServiceRegistry_ESModuleConsumesInfo_h

/**\class edm::ESModuleConsumesInfo

   Description: Contains information about a product
   a module declares it will consume from the EventSetup.

   Normally, there will be one object per consumes call.
   But in the case of a "may consumes", there will be multiple
   objects with each object corresponding to one of the products
   that might be consumed.

   Usage: These are typically available from the PathsAndConsumesOfModules
   object passed as an argument to a Service callback.
*/
//
// Original Author: W. David Dagenhart
//         Created: 11/14/2024

#include <string_view>

namespace edm {
  struct ESModuleConsumesInfo {
    ESModuleConsumesInfo();

    std::string_view moduleBaseType() const;

    // These are created by ESProducer.

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
    unsigned int produceMethodIDOfConsumer_;
    unsigned int produceMethodIDOfProducer_;
    bool isSource_;
    bool isLooper_;
    bool moduleLabelMismatch_;
    bool mayConsumes_;
    bool mayConsumesFirstEntry_;
    bool mayConsumesNoProducts_;
  };
}  // namespace edm
#endif
