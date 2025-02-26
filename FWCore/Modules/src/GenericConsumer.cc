/*
 *  This EDAnalyzer will depend on all the event, lumi, run or process products declared by its configuration, both
 *  transient and persistent.
 *
 *  The dependencies can be specified either as module labels (e.g. "<module label>") or as branch names (e.g. 
 *  "<product type>_<module label>_<instance name>_<process name>").
 *  If a module label is used, no underscore ("_") must be present; this module will depend all the products produced
 *  by that module, including those produced by the Transformer functionality (such as the implicitly copied-to-host
 *  products in case of Alpaka-based modules).
 *  If a branch name is used, all four fields must be present, separated by underscores; this module will depend only
 *  on the matching product(s).
 *  
 *  Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names,
 *  similar to an OutputModule's "keep" statements.
 *  Use "*" to depend on all products of a given category.
 *
 *  For example, in the case of Alpaka-based modules running on a device, using
 *
 *      eventProducts = cms.untracked.vstring( "module" )
 *
 *  will cause "module" to run, along with automatic copy of its device products to the host.
 *  To avoid the copy, the DeviceProduct branch can be specified explicitly with
 *
 *      eventProducts = cms.untracked.vstring( "*DeviceProduct_module_*_*" )
 *
 *  .
 */

#include <algorithm>
#include <string>
#include <regex>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class GenericConsumer : public edm::global::EDAnalyzer<> {
  public:
    explicit GenericConsumer(ParameterSet const&);
    ~GenericConsumer() override = default;

    void analyze(StreamID, Event const&, EventSetup const&) const override {}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    std::vector<edm::ProductNamePattern> eventProducts_;
    std::vector<edm::ProductNamePattern> lumiProducts_;
    std::vector<edm::ProductNamePattern> runProducts_;
    std::vector<edm::ProductNamePattern> processProducts_;
    std::string label_;
    bool verbose_;
  };

  GenericConsumer::GenericConsumer(ParameterSet const& config)
      : eventProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("eventProducts"))),
        lumiProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("lumiProducts"))),
        runProducts_(edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("runProducts"))),
        processProducts_(
            edm::productPatterns(config.getUntrackedParameter<std::vector<std::string>>("processProducts"))),
        label_(config.getParameter<std::string>("@module_label")),
        verbose_(config.getUntrackedParameter<bool>("verbose")) {
    callWhenNewProductsRegistered([this](edm::ProductDescription const& branch) {
      static const std::string kPathStatus("edm::PathStatus");
      static const std::string kEndPathStatus("edm::EndPathStatus");

      switch (branch.branchType()) {
        case InEvent:
          if (branch.className() == kPathStatus or branch.className() == kEndPathStatus)
            return;
          for (auto const& label : eventProducts_)
            if (label.match(branch)) {
              this->consumes(edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                             edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
              if (verbose_) {
                edm::LogVerbatim("GenericConsumer")
                    << label_ << " consumes Event product " << branch.friendlyClassName() << '_' << branch.moduleLabel()
                    << '_' << branch.productInstanceName() << '_' << branch.processName() << '\n';
              }
              break;
            }
          break;

        case InLumi:
          for (auto const& label : lumiProducts_)
            if (label.match(branch)) {
              this->consumes<edm::InLumi>(
                  edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                  edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
              if (verbose_) {
                edm::LogVerbatim("GenericConsumer")
                    << label_ << " consumes LuminosityBlock product " << branch.friendlyClassName() << '_'
                    << branch.moduleLabel() << '_' << branch.productInstanceName() << '_' << branch.processName()
                    << '\n';
              }
              break;
            }
          break;

        case InRun:
          for (auto const& label : runProducts_)
            if (label.match(branch)) {
              this->consumes<edm::InRun>(
                  edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                  edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
              if (verbose_) {
                edm::LogVerbatim("GenericConsumer")
                    << label_ << " consumes Run product " << branch.friendlyClassName() << '_' << branch.moduleLabel()
                    << '_' << branch.productInstanceName() << '_' << branch.processName() << '\n';
              }
              break;
            }
          break;

        case InProcess:
          for (auto const& label : processProducts_)
            if (label.match(branch)) {
              this->consumes<edm::InProcess>(
                  edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                  edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
              if (verbose_) {
                edm::LogVerbatim("GenericConsumer")
                    << label_ << " consumes Process product " << branch.friendlyClassName() << '_'
                    << branch.moduleLabel() << '_' << branch.productInstanceName() << '_' << branch.processName()
                    << '\n';
              }
              break;
            }
          break;

        default:
          throw Exception(errors::LogicError)
              << "Unexpected branch type " << branch.branchType() << "\nPlease contact a Framework developer\n";
      }
    });
  }

  void GenericConsumer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        R"(This EDAnalyzer will depend on all the event, lumi, run or process products declared by its configuration, both transient and persistent.

The dependencies can be specified either as module labels (e.g. "<module label>") or as branch names (e.g. "<product type>_<module label>_<instance name>_<process name>").
If a module label is used, no underscore ("_") must be present; this module will depend all the products produced by that module, including those produced by the Transformer functionality (such as the implicitly copied-to-host products in case of Alpaka-based modules).
If a branch name is used, all four fields must be present, separated by underscores; this module will depend only on the matching product(s).

Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names, similar to an OutputModule's "keep" statements.
Use "*" to depend on all products of a given category.

For example, in the case of Alpaka-based modules running on a device, using

    eventProducts = cms.untracked.vstring( "module" )

will cause "module" to run, along with automatic copy of its device products to the host.
To avoid the copy, the DeviceProduct branch can be specified explicitly with

    eventProducts = cms.untracked.vstring( "*DeviceProduct_module_*_*" )

.)");

    ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("eventProducts", {})
        ->setComment("List of modules or branches whose event products this module will depend on.");
    desc.addUntracked<std::vector<std::string>>("lumiProducts", {})
        ->setComment("List of modules or branches whose lumi products this module will depend on.");
    desc.addUntracked<std::vector<std::string>>("runProducts", {})
        ->setComment("List of modules or branches whose run products this module will depend on.");
    desc.addUntracked<std::vector<std::string>>("processProducts", {})
        ->setComment("List of modules or branches whose process products this module will depend on.");
    desc.addUntracked<bool>("verbose", false)
        ->setComment("Print the actual branch names for which the dependency are declared.");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::GenericConsumer;
DEFINE_FWK_MODULE(GenericConsumer);
