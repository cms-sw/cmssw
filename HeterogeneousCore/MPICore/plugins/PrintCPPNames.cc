#include <fstream>
#include <vector>
#include <string>

// JSON headers
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// CMSSW
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class PrintCPPNames : public edm::global::EDAnalyzer<> {
public:
  explicit PrintCPPNames(edm::ParameterSet const& pset)
      : outputFile_(pset.getParameter<std::string>("outputFile")), products_(json::array()) {
    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static constexpr std::string_view kPathStatus("edm::PathStatus");
      static constexpr std::string_view kEndPathStatus("edm::EndPathStatus");

      if (product.className() == kPathStatus || product.className() == kEndPathStatus)
        return;

      products_.push_back({{"instance", product.friendlyClassName()},
                           {"module", product.moduleLabel()},
                           {"product_instance", product.productInstanceName()},
                           {"process", product.processName()},
                           {"type", product.unwrappedType().name()},
                           {"branch", product.branchType()}});
    });
  }

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {}

  void endJob() override {
    std::ofstream out(outputFile_);
    out << products_.dump(2);  // pretty-print with indent 2
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("outputFile", "print_cppnames.json");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::string outputFile_;
  json products_;  // JSON array for product infos
};

DEFINE_FWK_MODULE(PrintCPPNames);
