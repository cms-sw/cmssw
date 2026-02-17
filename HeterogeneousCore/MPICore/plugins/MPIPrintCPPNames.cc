#include <iostream>
#include <string_view>

// CMSSW
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

class PrintCPPNames : public edm::stream::EDProducer<> {
public:
  explicit PrintCPPNames(edm::ParameterSet const&) {
    callWhenNewProductsRegistered([](edm::ProductDescription const& product) {
      static constexpr std::string_view kPathStatus("edm::PathStatus");
      static constexpr std::string_view kEndPathStatus("edm::EndPathStatus");

      // Skip framework-internal status products if desired
      if (product.className() == kPathStatus || product.className() == kEndPathStatus)
        return;

      std::cout << "PrintCPPNames: considering product " << product.friendlyClassName() << '_' << product.moduleLabel()
                << '_' << product.productInstanceName() << '_' << product.processName() << " of type "
                << product.unwrappedType().name() << " branch type " << product.branchType() << '\n';
    });
  }

  void produce(edm::Event&, edm::EventSetup const&) override {
    // intentionally empty
  }
};

DEFINE_FWK_MODULE(PrintCPPNames);
