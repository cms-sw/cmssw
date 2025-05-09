/*
 *  This EDAnalyzer will query the TClass, ClassProperty and ClassInfo of all the persistent products specified in its
 *  configuration.
 *
 *  The products can be specified either as module labels (e.g. "<module label>") or as full product names (e.g. 
 *  "<product type>_<module label>_<instance name>_<process name>").
 *  If a module label is used, no underscore ("_") must be present; this module will check the types of all the
 *  collections produced by that module, including those produced by the Transformer functionality (such as the
 *  implicitly copied-to-host products in case of Alpaka-based modules).
 *  If a full product name is used, all four fields must be present, separated by underscores; this module will depend
 *  only on the matching product(s).
 *  
 *  Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names,
 *  similar to an OutputModule's "keep" statements.
 *  Use "*" to check all products of a given category.
 */

#include <algorithm>
#include <string>
#include <regex>
#include <vector>

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edmtest {

  class CheckClassInfo : public edm::global::EDAnalyzer<> {
  public:
    explicit CheckClassInfo(edm::ParameterSet const&);
    ~CheckClassInfo() override = default;

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void check(edm::ProductDescription const& product) const;

    std::vector<edm::ProductNamePattern> eventProducts_;
    std::vector<edm::ProductNamePattern> lumiProducts_;
    std::vector<edm::ProductNamePattern> runProducts_;
    std::vector<edm::ProductNamePattern> processProducts_;
  };

  CheckClassInfo::CheckClassInfo(edm::ParameterSet const& config)
      : eventProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("eventProducts"))),
        lumiProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("lumiProducts"))),
        runProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("runProducts"))),
        processProducts_(
            edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("processProducts"))) {
    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      switch (product.branchType()) {
        case edm::InEvent:
          for (auto const& label : eventProducts_)
            if (label.match(product)) {
              check(product);
              break;
            }
          break;

        case edm::InLumi:
          for (auto const& label : lumiProducts_)
            if (label.match(product)) {
              check(product);
              break;
            }
          break;

        case edm::InRun:
          for (auto const& label : runProducts_)
            if (label.match(product)) {
              check(product);
              break;
            }
          break;

        case edm::InProcess:
          for (auto const& label : processProducts_)
            if (label.match(product)) {
              check(product);
              break;
            }
          break;

        default:
          throw edm::Exception(edm::errors::LogicError)
              << "Unexpected branch type " << product.branchType() << "\nPlease contact a Framework developer\n";
      }
    });
  }

  void CheckClassInfo::check(edm::ProductDescription const& product) const {
    if (product.transient()) {
      edm::LogVerbatim("CheckClassInfo") << "The product " << product.friendlyClassName() << '_'
                                         << product.moduleLabel() << '_' << product.productInstanceName() << '_'
                                         << product.processName() << " is transient, and will not be queried.";
    } else {
      edm::LogVerbatim("CheckClassInfo") << "The product " << product.friendlyClassName() << '_'
                                         << product.moduleLabel() << '_' << product.productInstanceName() << '_'
                                         << product.processName() << " has:\n  wrapper type "
                                         << product.wrappedType().name();
      TClass* type = product.wrappedType().getClass();
      edm::LogVerbatim("CheckClassInfo") << "  TClass pointer " << type;
      if (type == nullptr) {
        throw edm::Exception(edm::errors::DictionaryNotFound)
            << "Failed to get a valid TClass pointer for the persistent type " << product.wrappedType().name();
      }
      auto prop = type->ClassProperty();
      edm::LogVerbatim("CheckClassInfo") << "  TClass property " << prop;
      ClassInfo_t* info = type->GetClassInfo();
      edm::LogVerbatim("CheckClassInfo") << "  ClassInfo pointer " << info;
      if (info == nullptr) {
        throw edm::Exception(edm::errors::DictionaryNotFound)
            << "Failed to get a valid ClassInfo_t pointer for the persistent type " << product.wrappedType().name();
      }
    }
    edm::LogVerbatim("CheckClassInfo");  // print an empty line between products
  }

  void CheckClassInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        R"(This EDAnalyzer will query the TClass, ClassProperty and ClassInfo of all the persistent products specified in its
configuration.

The products can be specified either as module labels (e.g. "<module label>") or as full product names (e.g.
"<product type>_<module label>_<instance name>_<process name>").
If a module label is used, no underscore ("_") must be present; this module will check the types of all the collections
produced by that module, including those produced by the Transformer functionality (such as the implicitly copied-to-host
products in case of Alpaka-based modules).
If a full product name is used, all four fields must be present, separated by underscores; this module will depend only
on the matching product(s).

Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names, similar
to an OutputModule's "keep" statements. Use "*" to check all products of a given category.)");

    edm::ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("eventProducts", {})
        ->setComment("List of modules or product names whose event products this module will check.");
    desc.addUntracked<std::vector<std::string>>("lumiProducts", {})
        ->setComment("List of modules or product names whose lumi products this module will check.");
    desc.addUntracked<std::vector<std::string>>("runProducts", {})
        ->setComment("List of modules or product names whose run products this module will check.");
    desc.addUntracked<std::vector<std::string>>("processProducts", {})
        ->setComment("List of modules or product names whose process products this module will check.");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::CheckClassInfo);
