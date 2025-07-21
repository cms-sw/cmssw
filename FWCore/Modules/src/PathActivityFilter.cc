#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/PathActivityToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class PathActivityFilter : public global::EDFilter<> {
  public:
    explicit PathActivityFilter(ParameterSet const& config)
        : token_(consumes(config.getParameter<edm::InputTag>("producer"))) {}

    bool filter(StreamID sid, Event& event, EventSetup const& setup) const final {
      return event.getHandle(token_).isValid();
    }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const edm::EDGetTokenT<PathActivityToken> token_;
  };

  void PathActivityFilter::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("producer", {"pathActivityProducer"});
    descriptions.addWithDefaultLabel(desc);
    descriptions.setComment(
        "This EDFilter tries to consume an edm::PathActivityToken: if it finds it, it returns \"true\", otherwise "
        "\"false\".");
  }
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::PathActivityFilter;
DEFINE_FWK_MODULE(PathActivityFilter);
