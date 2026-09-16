#include <fstream>
#include <map>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

// JSON headers
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// CMSSW
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"

class DumpProductNames : public edm::global::EDAnalyzer<> {
public:
  explicit DumpProductNames(edm::ParameterSet const& pset)
      : outputFile_(pset.getParameter<std::string>("outputFile")), products_(json::array()) {
    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      // For each data product, store the following fields
      json entry = {{"friendly_type_name", product.friendlyClassName()},
                    {"module", product.moduleLabel()},
                    {"product_instance", product.productInstanceName()},
                    {"process", product.processName()},
                    {"type", product.unwrappedType().name()},
                    {"branch", product.branchType()}};
      if (auto const& types = viewTypes(product); not types.empty()) {
        entry["view_types"] = types;
      }
      if (product.isAlias()) {
        // For EDAliases, add the module and product instance of the original
        // data product
        if (auto it = originals_.find(product.originalBranchID()); it != originals_.end()) {
          entry["alias_for"] = {{"module", it->second.first}, {"product_instance", it->second.second}};
        }
      } else {
        originals_.emplace(product.branchID(), std::pair(product.moduleLabel(), product.productInstanceName()));
      }
      products_.push_back(std::move(entry));
    });
  }

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {}

  void endJob() override {
    std::ofstream out(outputFile_);
    out << products_.dump(2);  // pretty-print with indent 2
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "This modul dumps a file with C++ names and some other data about the products in a process."
        "It produces a json file, which is a list of entries for each product registered by the framweork."
        " Each entry contains following fields:"
        "\n - friendly_type_name - friendly type name"
        "\n - module - name of the module which produces the registered product"
        "\n - product_instance - if available"
        "\n - process - name of the process which produces the registered product"
        "\n - type - C++ type"
        "\n - branch "
        "\n - alias_for - for an EDAlias, the module and product_instance of the original product"
        "\n - view_types - for a container, the friendly names of the element types T for which an"
        " edm::View<T> consumer can read it: its element type and the public base classes of that type");
    desc.add<std::string>("outputFile", "print_cppnames.json");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  std::vector<std::string> const& viewTypes(edm::ProductDescription const& product) {
    // Return the friendly names of the element types T for which an
    // edm::View<T> can read the data product. The list is empty when no
    // edm::View can read the data product.
    auto [it, inserted] = viewTypes_.try_emplace(product.className());
    if (not inserted or not product.wrappedType()) {
      // already computed for this type, or the type has no ROOT dictionary
      return it->second;
    }
    edm::TypeID const elementType = edm::productholderindexhelper::getContainedTypeFromWrapper(
        edm::TypeID(product.wrappedType().typeInfo()), product.className());
    if (elementType == edm::TypeID(typeid(void)) or elementType == edm::TypeID()) {
      // not a container, so no edm::View can read it
      return it->second;
    }

    it->second.push_back(elementType.friendlyClassName());

    // add the base classes of the element type: an edm::View<Base> can read a container of Derived
    std::vector<std::string> missingDictionaries;
    std::vector<edm::TypeID> baseTypes;
    edm::public_base_classes(missingDictionaries, elementType, baseTypes);
    for (edm::TypeID const& base : baseTypes) {
      it->second.push_back(base.friendlyClassName());
    }
    return it->second;
  }

  // module label and product instance name of each data product that is not an EDAlias
  std::map<edm::BranchID, std::pair<std::string, std::string>> originals_;
  // the view_types of each C++ type, by class name
  std::map<std::string, std::vector<std::string>> viewTypes_;
  std::string outputFile_;
  json products_;  // JSON array for product infos
};

DEFINE_FWK_MODULE(DumpProductNames);
