#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {
  /**
   * This class is the physical EDProducer part of the SwitchProducer
   * infrastructure. It must be configured only with the
   * SwitchProducer python construct.
   *
   * The purposes of this EDProducer are
   * - Create the consumes() links to the chosen case to make the prefetching work correclty
   * - Forward the produces() information to create proper BranchDescription objects
   */
  class SwitchProducer: public global::EDProducer<> {
  public:
    explicit SwitchProducer(ParameterSet const& iConfig);
    ~SwitchProducer() override = default;
     static void fillDescriptions(ConfigurationDescriptions& descriptions);
    void produce(StreamID, Event& e, EventSetup const& c) const final {}
  };

  SwitchProducer::SwitchProducer(ParameterSet const& iConfig) {
    auto const& moduleLabel = iConfig.getParameter<std::string>("@module_label");
    auto const& chosenLabel = iConfig.getUntrackedParameter<std::string>("@chosen_case");
    callWhenNewProductsRegistered([=](edm::BranchDescription const& iBranch) {
        if(iBranch.moduleLabel() == chosenLabel) {
          if(iBranch.branchType() != InEvent) {
            throw Exception(errors::UnimplementedFeature) << "SwitchProducer does not support non-event branches. Got " << iBranch.branchType() << " for SwitchProducer with label " << moduleLabel << " whose chosen case is " << chosenLabel << ".";
          }

          // With consumes, create the connection to the chosen case EDProducer for prefetching
          this->consumes(edm::TypeToGet{iBranch.unwrappedTypeID(),PRODUCT_TYPE},
                         edm::InputTag{iBranch.moduleLabel(), iBranch.productInstanceName(), iBranch.processName()});
          // With produces, create a producer-like BranchDescription
          // early-enough for it to be flagged as non-OnDemand in case
          // the SwithcProducer is on a Path
          this->produces(iBranch.unwrappedTypeID(), iBranch.productInstanceName()).setSwitchAlias(iBranch.moduleLabel());
        }
      });
  }

  void SwitchProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.add<std::vector<std::string>>("@all_cases");
    desc.addUntracked<std::string>("@chosen_case");
    descriptions.addDefault(desc);
  }
}

using edm::SwitchProducer;
DEFINE_FWK_MODULE(SwitchProducer);
