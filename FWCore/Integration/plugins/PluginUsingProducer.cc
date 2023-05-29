/** \class edmtest::PluginUsingProducer
\author Chris Jones, created 21 September 2018
*/

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"

#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {
  struct IntMakerBase {
    virtual ~IntMakerBase() = default;
    virtual int value() const = 0;
  };

  using IntFactory = edmplugin::PluginFactory<IntMakerBase*(edm::ParameterSet const&)>;
}  // namespace edmtest
EDM_REGISTER_VALIDATED_PLUGINFACTORY(edmtest::IntFactory, "edmtestIntFactory");

namespace edmtest {

  struct OneMaker : public IntMakerBase {
    explicit OneMaker(edm::ParameterSet const&) {}
    int value() const final { return 1; };

    static void fillPSetDescription(edm::ParameterSetDescription&) {}
  };

  struct ValueMaker : public IntMakerBase {
    explicit ValueMaker(edm::ParameterSet const& iPSet) : value_{iPSet.getParameter<int>("value")} {}
    int value() const final { return value_; };

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) { iDesc.add<int>("value", 5); }

    int value_;
  };

  class PluginUsingProducer : public edm::global::EDProducer<> {
  public:
    explicit PluginUsingProducer(edm::ParameterSet const&);

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& iConf) {
      edm::ParameterSetDescription pluginDesc;
      pluginDesc.addNode(edm::PluginDescription<IntFactory>("type", "edmtestValueMaker", true));

      edm::ParameterSetDescription top;
      top.add<edm::ParameterSetDescription>("plugin", pluginDesc);

      iConf.addWithDefaultLabel(top);
    }

  private:
    std::unique_ptr<IntMakerBase> maker_;
    edm::EDPutTokenT<int> putToken_;
  };

  PluginUsingProducer::PluginUsingProducer(edm::ParameterSet const& pset) : putToken_{produces<int>()} {
    auto pluginPSet = pset.getParameter<edm::ParameterSet>("plugin");
    maker_ = std::unique_ptr<IntMakerBase>{
        IntFactory::get()->create(pluginPSet.getParameter<std::string>("type"), pluginPSet)};
  }

  void PluginUsingProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
    event.emplace(putToken_, maker_->value());
  }
}  // namespace edmtest
using edmtest::PluginUsingProducer;
DEFINE_FWK_MODULE(PluginUsingProducer);

DEFINE_EDM_VALIDATED_PLUGIN(edmtest::IntFactory, edmtest::OneMaker, "edmtestOneMaker");
DEFINE_EDM_VALIDATED_PLUGIN(edmtest::IntFactory, edmtest::ValueMaker, "edmtestValueMaker");
