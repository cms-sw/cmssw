#include "FWCore/Framework/interface/NoProductResolverException.h"

#include <cstring>

namespace edm {
  namespace eventsetup {

    NoProductResolverException::NoProductResolverException(const EventSetupRecordKey& key,
                                                           const char* iClassName,
                                                           const char* productLabel,
                                                           bool moduleLabelDoesNotMatch)
        : cms::Exception("NoProductResolverException") {
      append("Cannot find EventSetup module to produce data of type \"");
      append(iClassName);
      append("\" in\nrecord \"");
      append(key.name());
      if (std::strncmp(productLabel, "@mayConsume", 11) == 0) {
        // I'd rather print out the product label here, but we don't know
        // it at this point for the "may consume" case.
        append("\", which is being consumed via a call to setMayConsume.\n");
      } else {
        append("\" with product label \"");
        append(productLabel);
        append("\".\n");
      }
      if (moduleLabelDoesNotMatch) {
        // We discussed adding the requested and preferred module label into
        // the following message. That information is not currently available
        // and it would require some extra memory resources to make it available.
        // Given how rarely this exception occurs and how little used the feature
        // of selecting module labels is, we decided that it was not worth the
        // the extra memory. It could be done and if something changes and we end
        // up debugging these errors in the future often, it might be worth adding
        // the extra info...
        append(
            "An ESSource or ESProducer is configured to produce this data, but the\n"
            "ESInputTag in the configuration also requires a specific module label.\n"
            "The preferred module to produce this data in the configuration has a\n"
            "different module label.\n");
        append(
            "Please ensure that there is an ESSource or ESProducer configured with\n"
            "the requested module label that both produces this data and is selected\n"
            "as the preferred provider for this data. If there is more than one such\n"
            "ESSource or ESProducer you may need to use an ESPrefer directive in the\n"
            "configuration.");
      } else {
        append("Please add an ESSource or ESProducer to your job which can deliver this data.\n");
      }
    }
  }  // namespace eventsetup
}  // namespace edm
