#include "FWCore/Sources/interface/PuttableSourceBase.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <memory>
namespace edmtest {
  namespace {
    edm::ProductDescription makeDesc(std::string const& iProcess) {
      edm::ProductDescription desc(
          edm::InEvent, "TriggerResults", iProcess, "", edm::TypeID(typeid(edm::TriggerResults)));
      desc.setIsProvenanceSetOnRead();
      desc.setProduced(false);
      return desc;
    }

    std::pair<edm::ParameterSetID, edm::ParameterSetID> calcID(edm::ProductDescription const& iDesc,
                                                               std::vector<std::string> const& iPaths) {
      edm::ParameterSet pset;
      std::string const& processName = iDesc.processName();
      typedef std::vector<std::string> vstring;
      vstring empty;

      vstring modlbl;
      pset.addParameter("@all_sources", modlbl);

      edm::ParameterSet triggerPaths;
      triggerPaths.addParameter<vstring>("@trigger_paths", iPaths);
      pset.addParameter<edm::ParameterSet>("@trigger_paths", triggerPaths);
      triggerPaths.registerIt();

      pset.addParameter<vstring>("@all_esmodules", empty);
      pset.addParameter<vstring>("@all_esprefers", empty);
      pset.addParameter<vstring>("@all_essources", empty);
      pset.addParameter<vstring>("@all_loopers", empty);
      pset.addParameter<vstring>("@all_modules", empty);
      pset.addParameter<vstring>("@end_paths", empty);
      pset.addParameter<vstring>("@paths", iPaths);
      pset.addParameter<std::string>("@process_name", processName);
      // Now we register the process parameter set.
      pset.registerIt();

      return std::make_pair(pset.id(), triggerPaths.id());
    }
  }  // namespace
  class TriggerResultsTestSource : public edm::IDGeneratorSourceBase<edm::PuttableSourceBase> {
  public:
    TriggerResultsTestSource(edm::ParameterSet const&, edm::InputSourceDescription const&);

    void readEvent_(edm::EventPrincipal& eventPrincipal) final;

    bool setRunAndEventInfo(edm::EventID& id,
                            edm::TimeValue_t& time,
                            edm::EventAuxiliary::ExperimentType& etype) override {
      return true;
    }
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const std::string process_;
    const std::vector<std::string> paths_;
    const std::vector<unsigned int> states_;
    const edm::ProductDescription desc_;
    edm::ParameterSetID processPsetID_;
    edm::ParameterSetID psetID_;
  };

  TriggerResultsTestSource::TriggerResultsTestSource(edm::ParameterSet const& iPSet,
                                                     edm::InputSourceDescription const& iDesc)
      : edm::IDGeneratorSourceBase<edm::PuttableSourceBase>(iPSet, iDesc, false),
        process_(iPSet.getUntrackedParameter<std::string>("process")),
        paths_(iPSet.getUntrackedParameter<std::vector<std::string>>("paths")),
        states_(iPSet.getUntrackedParameter<std::vector<unsigned int>>("pathStates")),
        desc_(makeDesc(process_)) {
    {
      auto [processPsetID, psetID] = calcID(desc_, paths_);
      processPsetID_ = std::move(processPsetID);
      psetID_ = std::move(psetID);
    }
    std::vector<edm::BranchDescription> products;
    products.reserve(1);
    products.emplace_back(desc_);
    std::vector<std::string> processOrder(1, process_);
    productRegistryUpdate().updateFromInput(products, processOrder);
    assert(paths_.size() == states_.size());

    edm::ProcessHistory ph;
    ph.emplace_back(process_, processPsetID_, edm::getReleaseVersion(), edm::HardwareResourcesDescription());
    processHistoryRegistryForUpdate().registerProcessHistory(ph);
  }

  void TriggerResultsTestSource::fillDescriptions(edm::ConfigurationDescriptions& iConfigs) {
    edm::ParameterSetDescription desc;
    edm::IDGeneratorSourceBase<edm::PuttableSourceBase>::fillDescription(desc);
    desc.addUntracked<std::string>("process", std::string(""));
    desc.addUntracked<std::vector<std::string>>("paths")->setComment("names of paths");
    desc.addUntracked<std::vector<unsigned int>>("pathStates")
        ->setComment(
            "The state of the path. The number of entires must be the same as in 'paths' and the value must be between "
            "0-2 inclusive.");

    iConfigs.addDefault(desc);
  }

  void TriggerResultsTestSource::readEvent_(edm::EventPrincipal& eventPrincipal) {
    doReadEvent(eventPrincipal, [this](edm::EventPrincipal& eventPrincipal) {
      edm::HLTGlobalStatus status(paths_.size());
      for (std::size_t i = 0; i < paths_.size(); ++i) {
        status.at(i) = edm::HLTPathStatus(static_cast<edm::hlt::HLTState>(states_[i]));
      }

      auto ptr = std::make_unique<edm::Wrapper<edm::TriggerResults>>(edm::WrapperBase::Emplace(), status, psetID_);
      eventPrincipal.put(desc_, std::move(ptr), edm::ProductProvenance(desc_.branchID()));
    });
  }
}  // namespace edmtest

using edmtest::TriggerResultsTestSource;
DEFINE_FWK_INPUT_SOURCE(TriggerResultsTestSource);
