#include <optional>
#include <utility>

#include "DataFormats/HeterogeneousTutorial/interface/alpaka/JetsDeviceCollection.h"
#include "DataFormats/HeterogeneousTutorial/interface/alpaka/TripletsDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/Tutorial/interface/alpaka/JetsSelectionDeviceCollection.h"

#include "InvariantMassSelectorAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  class InvariantMassSelector : public stream::SynchronizingEDProducer<> {
  public:
    InvariantMassSelector(edm::ParameterSet const& config)
        : SynchronizingEDProducer<>(config),
          jets_{consumes(config.getParameter<edm::InputTag>("jets"))},
          ntuplets_{produces()},
          size_{cms::alpakatools::make_host_buffer<int32_t, Platform>()},
          cuts_{PortableHostObject<InvariantMassSelection>{
              cms::alpakatools::host(),
              InvariantMassSelection{static_cast<float>(config.getParameter<double>("pT_min")),
                                     static_cast<float>(config.getParameter<double>("pT_max")),
                                     static_cast<float>(config.getParameter<double>("eta_min")),
                                     static_cast<float>(config.getParameter<double>("eta_max")),
                                     static_cast<float>(config.getParameter<double>("mass_min")),
                                     static_cast<float>(config.getParameter<double>("mass_max"))}}}  //
    {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("jets");
      desc.add<double>("pT_min")->setComment("Minimum pT of the input jets");
      desc.add<double>("pT_max")->setComment("Maximum pT of the input jets");
      desc.add<double>("eta_min")->setComment("Minimum |eta| of the input jets");
      desc.add<double>("eta_max")->setComment("Maximum |eta| of the input jets");
      desc.add<double>("mass_min")->setComment("Minimum invariant mass of the jet ntuplets");
      desc.add<double>("mass_max")->setComment("Maximum invariant mass of the jets ntuplets");

      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& event, device::EventSetup const&) override {
      auto const& jets = event.get(jets_);

      // Initialise the variables use to hold the results of the asynchronous code
      *size_ = 0;
      selection_.emplace(jets.view().metadata().size(), event.queue());

      // Apply the selection cuts and count how many jets pass the selection.
      InvariantMassSelection const* cuts = cuts_.get(event.queue()).data();
      InvariantMassSelectorAlgo::applySelections(event.queue(), cuts, jets.view(), selection_->view(), size_.data());
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      auto const& jets = event.get(jets_);

      // Once the framework calls produce() it is guaranteed that the size_ has been filled in host memory by the device:
      //   - the number of jets passing the cuts is N = size_
      //   - the number of potential doublets is (ᴺ₂) = (size_ × size_-1) / 2
      //   - the number of potential triplets is (ᴺ₃) = (size_ × size_-1 × size_-2) / 6
      // With up to 200 jets in an event, the result and intermediate steps should still fit in an int.
      int size = *size_;
      int combos = size * (size - 1) / 2 + size * (size - 1) * (size - 2) / 6;
      edm::LogInfo("InvariantMassSelector")
          << "Found " << size << " jets passing the selection cuts, will allocate space for " << combos << " ntuplets";
      TripletsDeviceCollection ntuplets(combos, event.queue());

      // Fill the JetsDeviceCollection with zeroes
      ntuplets.zeroInitialise(event.queue());

      // Find the pairs and triplets of jets with an invariant mass in the requested range.
      InvariantMassSelection const* cuts = cuts_.get(event.queue()).data();
      InvariantMassSelectorAlgo::findDoulets(event.queue(), cuts, jets.view(), selection_->view(), ntuplets.view());
      InvariantMassSelectorAlgo::findTriplets(event.queue(), cuts, jets.view(), selection_->view(), ntuplets.view());

      // Put the results into the Event.
      event.emplace(ntuplets_, std::move(ntuplets));
    }

  private:
    const device::EDGetToken<JetsDeviceCollection> jets_;
    const device::EDPutToken<TripletsDeviceCollection> ntuplets_;

    cms::alpakatools::host_buffer<int32_t> size_;
    std::optional<JetsSelectionDeviceCollection> selection_;

    cms::alpakatools::MoveToDeviceCache<Device, PortableHostObject<InvariantMassSelection>> cuts_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(tutorial::InvariantMassSelector);
