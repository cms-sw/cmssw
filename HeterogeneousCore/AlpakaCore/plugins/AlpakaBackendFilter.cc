#include <array>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"

class AlpakaBackendFilter : public edm::global::EDFilter<> {
public:
  explicit AlpakaBackendFilter(edm::ParameterSet const& config)
      : producer_(consumes<unsigned short>(config.getParameter<edm::InputTag>("producer"))), backends_{} {
    for (auto const& backend : config.getParameter<std::vector<std::string>>("backends")) {
      backends_[static_cast<unsigned short>(cms::alpakatools::toBackend(backend))] = true;
    }
  }

  bool filter(edm::StreamID sid, edm::Event& event, edm::EventSetup const& setup) const final {
    return backends_[event.get(producer_)];
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<unsigned short> producer_;
  std::array<bool, static_cast<short>(cms::alpakatools::Backend::size)> backends_;
};

void AlpakaBackendFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("producer", edm::InputTag("alpakaBackendProducer", "backend"))
      ->setComment(
          "Use the 'backend' instance label to read the backend indicator that is implicitly produced by every alpaka "
          "EDProducer.");
  desc.add<std::vector<std::string>>("backends", {"SerialSync"})
      ->setComment("Valid backends are 'SerialSync', 'CudaAsync', 'ROCmAsync', and 'TbbAsync'.");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "This EDFilter accepts events if the alpaka EDProducer 'producer' was run on a backend among those listed by the "
      "'backends' parameter.");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlpakaBackendFilter);
