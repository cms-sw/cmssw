#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalElectronicsMappingDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include <alpaka/alpaka.hpp>

#include "DeclsForKernels.h"
#include "UnpackPortable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalRawToDigiPortable : public stream::EDProducer<> {
  public:
    explicit EcalRawToDigiPortable(edm::ParameterSet const& ps);
    ~EcalRawToDigiPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    void produce(device::Event&, device::EventSetup const&) override;

  private:
    edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
    using OutputProduct = EcalDigiDeviceCollection;
    device::EDPutToken<OutputProduct> digisDevEBToken_;
    device::EDPutToken<OutputProduct> digisDevEEToken_;
    device::ESGetToken<EcalElectronicsMappingDevice, EcalMappingElectronicsRcd> eMappingToken_;

    std::vector<int> fedsToUnpack_;

    ecal::raw::ConfigurationParameters config_;
  };

  void EcalRawToDigiPortable::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
    std::vector<int> feds(54);
    for (uint32_t i = 0; i < 54; ++i)
      feds[i] = i + 601;
    desc.add<std::vector<int>>("FEDs", feds);
    desc.add<uint32_t>("maxChannelsEB", 61200);
    desc.add<uint32_t>("maxChannelsEE", 14648);
    desc.add<std::string>("digisLabelEB", "ebDigis");
    desc.add<std::string>("digisLabelEE", "eeDigis");

    confDesc.addWithDefaultLabel(desc);
  }

  EcalRawToDigiPortable::EcalRawToDigiPortable(const edm::ParameterSet& ps)
      : EDProducer(ps),
        rawDataToken_{consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("InputLabel"))},
        digisDevEBToken_{produces(ps.getParameter<std::string>("digisLabelEB"))},
        digisDevEEToken_{produces(ps.getParameter<std::string>("digisLabelEE"))},
        eMappingToken_{esConsumes()},
        fedsToUnpack_{ps.getParameter<std::vector<int>>("FEDs")} {
    config_.maxChannelsEB = ps.getParameter<uint32_t>("maxChannelsEB");
    config_.maxChannelsEE = ps.getParameter<uint32_t>("maxChannelsEE");
  }

  void EcalRawToDigiPortable::produce(device::Event& event, device::EventSetup const& setup) {
    // conditions
    auto const& eMappingProduct = setup.getData(eMappingToken_);

    // event data
    const auto rawDataHandle = event.getHandle(rawDataToken_);

    // make a first iteration over the FEDs to compute the total buffer size
    uint32_t size = 0;
    uint32_t feds = 0;
    for (auto const& fed : fedsToUnpack_) {
      auto const& data = rawDataHandle->FEDData(fed);
      auto const nbytes = data.size();

      // skip empty FEDs
      if (nbytes < globalFieds::EMPTYEVENTSIZE)
        continue;

      size += nbytes;
      ++feds;
    }

    auto& queue = event.queue();

    // input host buffers
    ecal::raw::InputDataHost inputHost(queue, size, feds);

    // output device collections
    OutputProduct digisDevEB{static_cast<int32_t>(config_.maxChannelsEB), queue};
    OutputProduct digisDevEE{static_cast<int32_t>(config_.maxChannelsEE), queue};
    // reset the size scalar of the SoA
    // memset takes an alpaka view that is created from the scalar in a view to the device collection
    auto digiViewEB = cms::alpakatools::make_device_view<uint32_t>(queue, digisDevEB.view().size());
    auto digiViewEE = cms::alpakatools::make_device_view<uint32_t>(queue, digisDevEE.view().size());
    alpaka::memset(queue, digiViewEB, 0);
    alpaka::memset(queue, digiViewEE, 0);

    // iterate over FEDs to fill the host buffer
    uint32_t currentCummOffset = 0;
    uint32_t fedCounter = 0;
    for (auto const& fed : fedsToUnpack_) {
      auto const& data = rawDataHandle->FEDData(fed);
      auto const nbytes = data.size();

      // skip empty FEDs
      if (nbytes < globalFieds::EMPTYEVENTSIZE)
        continue;

      // copy raw data into host buffer
      std::memcpy(inputHost.data.data() + currentCummOffset, data.data(), nbytes);
      // set the offset in bytes from the start
      inputHost.offsets[fedCounter] = currentCummOffset;
      inputHost.feds[fedCounter] = fed;

      // this is the current offset into the buffer
      currentCummOffset += nbytes;
      ++fedCounter;
    }
    assert(currentCummOffset == size);
    assert(fedCounter == feds);

    // unpack if at least one FED has data
    if (fedCounter > 0) {
      ecal::raw::unpackRaw(queue, inputHost, digisDevEB, digisDevEE, eMappingProduct, fedCounter, currentCummOffset);
    }

    event.emplace(digisDevEBToken_, std::move(digisDevEB));
    event.emplace(digisDevEEToken_, std::move(digisDevEE));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(EcalRawToDigiPortable);
