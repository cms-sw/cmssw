#include <utility>

#include "DataFormats/HeterogeneousTutorial/interface/alpaka/JetsDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/Tutorial/interface/PortableHostTable.h"
#include "HeterogeneousCore/Tutorial/interface/SoACorrectorRecord.h"
#include "HeterogeneousCore/Tutorial/interface/alpaka/PortableTable.h"

#include "PFJetsSoACorrectorAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  using namespace ::tutorial;

  class PFJetsSoACorrector : public global::EDProducer<> {
  public:
    PFJetsSoACorrector(edm::ParameterSet const& config)
        : EDProducer<>(config),
          uncorrected_{consumes(config.getParameter<edm::InputTag>("jets"))},
          corrected_{produces()},
          table_{esConsumes()}  //
    {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("jets");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const& setup) const override {
      // Get the corrections from the EventSetup.
      PortableTable const& correction = setup.getData(table_);

      // Get the uncorrected jets from the Event.
      JetsDeviceCollection const& uncorrected = event.get(uncorrected_);

      // Allocate a new SoA for the corrected jets.
      JetsDeviceCollection corrected(uncorrected.view().metadata().size(), event.queue());

      // Apply the corrections and fill the new SoA.
      PFJetsSoACorrectorAlgo::applyJetCorrections(
          event.queue(), uncorrected.view(), corrected.view(), correction.table());

      // Move the SoA with the corrected jets into the Event.
      event.emplace(corrected_, std::move(corrected));
    }

  private:
    const device::EDGetToken<JetsDeviceCollection> uncorrected_;
    const device::EDPutToken<JetsDeviceCollection> corrected_;
    const device::ESGetToken<PortableTable, SoACorrectorRecord> table_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(tutorial::PFJetsSoACorrector);
