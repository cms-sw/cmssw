#include <iostream>
#include <string_view>

// CMSSW
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class PrintCPPNames : public edm::global::EDAnalyzer<> {
public:
  explicit PrintCPPNames(edm::ParameterSet const&) {
    callWhenNewProductsRegistered([](edm::ProductDescription const& product) {
      static constexpr std::string_view kPathStatus("edm::PathStatus");
      static constexpr std::string_view kEndPathStatus("edm::EndPathStatus");

      if (product.className() == kPathStatus || product.className() == kEndPathStatus)
        return;

      std::cout << "PrintCPPNames: considering product " << product.friendlyClassName() << '_' << product.moduleLabel()
                << '_' << product.productInstanceName() << '_' << product.processName() << " of type "
                << product.unwrappedType().name() << " branch type " << product.branchType() << '\n';
    });
  }

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {
    // intentionally empty
  }
};

DEFINE_FWK_MODULE(PrintCPPNames);
