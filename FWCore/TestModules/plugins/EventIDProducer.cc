// CMSSW include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class EventIDProducer : public edm::global::EDProducer<> {
  public:
    EventIDProducer(edm::ParameterSet const& config) : token_(produces()) {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const final {
      event.emplace(token_, event.id());
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDPutTokenT<edm::EventID> token_;
  };

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::EventIDProducer);
