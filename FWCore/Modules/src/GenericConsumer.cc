/*
 *  This plugin depends on all the event, lumi and run products produced by the modules listed in its configuration:
 *    - eventProducts: depend on the event products from these modules
 *    - lumiProducts:  depend on the lumi products from these modules
 *    - runProducts:   depend on the run products from these modules
 *
 *  Use "*" to depend on all the products in a given branch.
 */

#include <algorithm>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
    std::vector<std::string> eventLabels_;
    std::vector<std::string> lumiLabels_;
    std::vector<std::string> runLabels_;
    std::vector<std::string> processLabels_;
  };

  GenericConsumer::GenericConsumer(ParameterSet const& config)
      : eventLabels_(config.getUntrackedParameter<std::vector<std::string>>("eventProducts")),
        lumiLabels_(config.getUntrackedParameter<std::vector<std::string>>("lumiProducts")),
        runLabels_(config.getUntrackedParameter<std::vector<std::string>>("runProducts")),
        processLabels_(config.getUntrackedParameter<std::vector<std::string>>("processProducts")) {
    std::sort(eventLabels_.begin(), eventLabels_.end());
    std::sort(lumiLabels_.begin(), lumiLabels_.end());
    std::sort(runLabels_.begin(), runLabels_.end());
    std::sort(processLabels_.begin(), processLabels_.end());

    callWhenNewProductsRegistered([this](edm::BranchDescription const& branch) {
      static const std::string kWildcard("*");
      static const std::string kPathStatus("edm::PathStatus");
      static const std::string kEndPathStatus("edm::EndPathStatus");

      switch (branch.branchType()) {
        case InEvent:
          if (std::binary_search(eventLabels_.begin(), eventLabels_.end(), branch.moduleLabel()) or
              (std::binary_search(eventLabels_.begin(), eventLabels_.end(), kWildcard) and
               branch.className() != kPathStatus and branch.className() != kEndPathStatus))
            this->consumes(edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                           edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
          break;

        case InLumi:
          if (std::binary_search(lumiLabels_.begin(), lumiLabels_.end(), branch.moduleLabel()) or
              std::binary_search(lumiLabels_.begin(), lumiLabels_.end(), kWildcard))
            this->consumes<edm::InLumi>(
                edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
          break;

        case InRun:
          if (std::binary_search(runLabels_.begin(), runLabels_.end(), branch.moduleLabel()) or
              std::binary_search(runLabels_.begin(), runLabels_.end(), kWildcard))
            this->consumes<edm::InRun>(
                edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
          break;

        case InProcess:
          if (std::binary_search(processLabels_.begin(), processLabels_.end(), branch.moduleLabel()) or
              std::binary_search(processLabels_.begin(), processLabels_.end(), kWildcard))
            this->consumes<edm::InProcess>(
                edm::TypeToGet{branch.unwrappedTypeID(), PRODUCT_TYPE},
                edm::InputTag{branch.moduleLabel(), branch.productInstanceName(), branch.processName()});
          break;
        default:
          throw Exception(errors::LogicError)
              << "Unexpected branch type " << branch.branchType() << "\nPlease contact a Framework developer\n";
      }
    });
  }

  void GenericConsumer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This plugin depends on all the event, lumi and run products "
        "produced by the modules listed in its configuration.");

    ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("eventProducts", {})
        ->setComment(
            "List of modules whose event products this module will depend on. "
            "Use \"*\" to depend on all event products.");
    desc.addUntracked<std::vector<std::string>>("lumiProducts", {})
        ->setComment(
            "List of modules whose lumi products this module will depend on. "
            "Use \"*\" to depend on all lumi products.");
    desc.addUntracked<std::vector<std::string>>("runProducts", {})
        ->setComment(
            "List of modules whose run products this module will depend on. "
            "Use \"*\" to depend on all run products.");
    desc.addUntracked<std::vector<std::string>>("processProducts", {})
        ->setComment(
            "List of modules whose process products this module will depend on. "
            "Use \"*\" to depend on all process products.");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace edm

using edm::GenericConsumer;
DEFINE_FWK_MODULE(GenericConsumer);
