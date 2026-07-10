#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class PathStateRelease : public global::EDFilter<> {
  public:
    explicit PathStateRelease(ParameterSet const& config)
        : token_(consumes(config.getParameter<edm::InputTag>("state"))) {}

    bool filter(StreamID sid, Event& event, EventSetup const& setup) const final {
      return event.getHandle(token_).isValid();
    }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const edm::EDGetTokenT<PathStateToken> token_;
  };

  void PathStateRelease::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("state", {"pathStateCapture"});
    descriptions.addWithDefaultLabel(desc);
    descriptions.setComment(
        "This EDFilter tries to consume an edm::PathStateToken: if it finds it, it returns \"true\", otherwise "
        "\"false\".");
  }
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::PathStateRelease;
DEFINE_FWK_MODULE(PathStateRelease);
