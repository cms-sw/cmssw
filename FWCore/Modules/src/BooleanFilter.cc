#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class BooleanFilter : public global::EDFilter<> {
  public:
    explicit BooleanFilter(ParameterSet const& config)
        : token_(consumes<bool>(config.getParameter<edm::InputTag>("src"))) {}

    bool filter(StreamID sid, Event& event, EventSetup const& setup) const final { return event.get(token_); }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const edm::EDGetTokenT<bool> token_;
  };

  void BooleanFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag());
    descriptions.addWithDefaultLabel(desc);
    descriptions.setComment("This EDFilter accepts or rejects events based on the boolean value read from \"src\".");
  }
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::BooleanFilter;
DEFINE_FWK_MODULE(BooleanFilter);
