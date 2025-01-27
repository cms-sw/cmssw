#ifndef FWCore_ServiceRegistry_EventSetupConsumesInfo_h
#define FWCore_ServiceRegistry_EventSetupConsumesInfo_h

/**\class edm::EventSetupConsumesInfo

   Description: Contains information about a product
   a module declares it will consume from the EventSetup.

   Normally, there will be one object per consumes call.
   But in the case of a "may consumes", there will be multiple
   objects with each object corresponding to one of the products
   that might be consumed.

   Usage: These are typically returned by the PathsAndConsumesOfModules
   object while EventProcessor::beginJob is running.
*/
//
// Original Author: W. David Dagenhart
//         Created: 11/14/2024

#include "FWCore/Utilities/interface/Transition.h"

#include <string_view>

namespace edm {
  struct EventSetupConsumesInfo {
  public:
    EventSetupConsumesInfo();

    std::string_view moduleBaseType() const;

    // These are created by EDConsumerBase or ESProducer.

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
    // This has the same value as the edm::Transition enum for the consumes items of
    // an ED module. For EventSetup modules, it counts the calls to setWhatProduced that
    // correspond to the functions that produce the data (starting the count at 0). Usually
    // there is only one such function with the name "produce", but the Framework allows
    // multiple such functions (at most one actually named "produce"). In the Framework,
    // dependences and  prefetching data are handled at the level of these functions and
    // not at the module level.
    unsigned int transitionOfConsumer_;
    // Similar to the previous but this is the transition number of the produce method that
    // creates the data product.
    unsigned int transitionOfProducer_;
    bool isSource_;
    bool isLooper_;
    bool moduleLabelMismatch_;
    bool mayConsumes_;
    bool mayConsumesFirstEntry_;
    bool mayConsumesNoProducts_;
  };
}  // namespace edm
#endif
