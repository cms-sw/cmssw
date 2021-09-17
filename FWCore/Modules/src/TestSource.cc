#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

#include <vector>
#include <utility>

namespace edm {
  class TestSource : public InputSource {
  public:
    explicit TestSource(ParameterSet const&, InputSourceDescription const&);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    ItemType getNextItemType() final;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() final;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() final;
    void readEvent_(EventPrincipal& eventPrincipal) final;

    std::vector<std::pair<ItemType, EventID>> m_transitions;
    std::vector<std::pair<ItemType, EventID>>::const_iterator m_nextTransition;

    static ItemType stringToType(const std::string&);
  };

  TestSource::TestSource(ParameterSet const& pset, InputSourceDescription const& desc) : InputSource(pset, desc) {
    for (auto const& p : pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("transitions")) {
      m_transitions.emplace_back(stringToType(p.getUntrackedParameter<std::string>("type")),
                                 p.getUntrackedParameter<EventID>("id"));
    }
    m_nextTransition = m_transitions.begin();
  }

  TestSource::ItemType TestSource::stringToType(const std::string& iTrans) {
    if (iTrans == "IsStop") {
      return IsStop;
    }
    if (iTrans == "IsFile") {
      return IsFile;
    }
    if (iTrans == "IsRun") {
      return IsRun;
    }
    if (iTrans == "IsLumi") {
      return IsLumi;
    }
    if (iTrans == "IsEvent") {
      return IsEvent;
    }
    if (iTrans == "IsSynchronize") {
      return IsSynchronize;
    }

    throw edm::Exception(errors::Configuration) << "Unknown transition type \'" << iTrans << "\'";

    return IsInvalid;
  }

  TestSource::ItemType TestSource::getNextItemType() {
    if (m_nextTransition == m_transitions.end()) {
      return IsStop;
    }
    auto trans = m_nextTransition->first;
    ++m_nextTransition;
    return trans;
  }

  std::shared_ptr<RunAuxiliary> TestSource::readRunAuxiliary_() {
    auto it = m_nextTransition;
    --it;
    return std::make_shared<RunAuxiliary>(it->second.run(), Timestamp(0), Timestamp(10));
  }

  std::shared_ptr<LuminosityBlockAuxiliary> TestSource::readLuminosityBlockAuxiliary_() {
    auto it = m_nextTransition;
    --it;
    return std::make_shared<LuminosityBlockAuxiliary>(
        it->second.run(), it->second.luminosityBlock(), Timestamp(0), Timestamp(10));
  }

  void TestSource::readEvent_(EventPrincipal& eventPrincipal) {
    auto it = m_nextTransition;
    --it;
    EventAuxiliary aux(it->second, processGUID(), Timestamp(0), false);
    auto history = processHistoryRegistry().getMapped(aux.processHistoryID());

    eventPrincipal.fillEventPrincipal(aux, history);
  }

  void TestSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Generates the specified sequence of transitions.");
    ProducerSourceBase::fillDescription(desc);

    ParameterSetDescription trans;
    trans.addUntracked<std::string>("type");
    trans.addUntracked<edm::EventID>("id");
    desc.addVPSetUntracked("transitions", trans, {{}});
    descriptions.add("source", desc);
  }
}  // namespace edm

using edm::TestSource;
DEFINE_FWK_INPUT_SOURCE(TestSource);
