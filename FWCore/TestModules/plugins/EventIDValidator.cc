// CMSSW include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmtest {

  class EventIDValidator : public edm::global::EDAnalyzer<> {
  public:
    EventIDValidator(edm::ParameterSet const& config)
        : token_(consumes(config.getUntrackedParameter<edm::InputTag>("source"))) {}

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const final {
      auto const& id = event.get(token_);
      if (id != event.id()) {
        throw cms::Exception("InvalidValue") << "EventIDValidator: found invalid input value\n"
                                             << id << "\nwhile expecting\n"
                                             << event.id();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked("source", edm::InputTag{"eventIDProducer", ""})
          ->setComment("EventID product to read from the event");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<edm::EventID> token_;
  };

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::EventIDValidator);
