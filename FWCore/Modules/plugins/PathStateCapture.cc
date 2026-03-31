#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class PathStateCapture : public global::EDProducer<> {
  public:
    explicit PathStateCapture(ParameterSet const& config) : token_(produces()) {}

    void produce(StreamID sid, Event& event, EventSetup const& setup) const final { event.emplace(token_); }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const edm::EDPutTokenT<PathStateToken> token_;
  };

  void PathStateCapture::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
    descriptions.setComment("This EDProducer produces an edm::PathStateToken.");
  }
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::PathStateCapture;
DEFINE_FWK_MODULE(PathStateCapture);
