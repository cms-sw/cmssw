#include <utility>

#include "DataFormats/HeterogeneousTutorial/interface/JetsHostCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  // Make the names from the top-level tutorial namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::tutorial namespace.
  using namespace ::tutorial;

  class PFJetsSoAProducer : public global::EDProducer<> {
  public:
    PFJetsSoAProducer(edm::ParameterSet const& config)
        : EDProducer<>(config), jets_{consumes(config.getParameter<edm::InputTag>("jets"))}, soa_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("jets");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      auto const& jets = event.get(jets_);
      JetsHostCollection soa(jets.size(), event.queue());
      for (size_t i = 0; i < jets.size(); ++i) {
        auto const& jet = jets[i];
        soa.view()[i] = {
            static_cast<float>(jet.pt()), static_cast<float>(jet.eta()), static_cast<float>(jet.phi()), pi_mass};
      }
      event.emplace(soa_, std::move(soa));
    }

  private:
    constexpr static float pi_mass = 0.13957039;  // GeV

    const edm::EDGetTokenT<reco::PFJetCollection> jets_;
    const edm::EDPutTokenT<JetsHostCollection> soa_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

// JetsDeviceCollection is not used explicitly, but this header is needed to let
// the framework implement the automatic copy from host to device.
#include "DataFormats/HeterogeneousTutorial/interface/alpaka/JetsDeviceCollection.h"

// Declare this class as an alpaka-based heterogeneous module.
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(tutorial::PFJetsSoAProducer);
