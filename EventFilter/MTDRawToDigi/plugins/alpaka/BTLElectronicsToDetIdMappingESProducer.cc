#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CondFormats/MTDObjects/interface/BTLReadoutMap.h"
#include "CondFormats/DataRecord/interface/BTLReadoutMapRcd.h"
#include "CondFormats/MTDObjects/interface/alpaka/BTLElectronicsToDetIdMappingDevice.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsIndexing.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the dense table denseChannelIndex(fedId, hsLinkId, eLinkId, channelId)
  // -> (rawId, valid, partnerChannelId, side),
  // used by BTLPairAndCountDigisKernel and BTLFillDigisKernel
  class BTLElectronicsToDetIdMappingESProducer : public ESProducer {
  public:
    explicit BTLElectronicsToDetIdMappingESProducer(const edm::ParameterSet& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      readoutMapToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<BTLElectronicsToDetIdMappingHost> produce(const BTLReadoutMapRcd& iRecord) {
      const BTLReadoutMap& readoutMap = iRecord.get(readoutMapToken_);

      auto product = std::make_unique<BTLElectronicsToDetIdMappingHost>(cms::alpakatools::host(),
                                                                        BTLElectronicsIndexing::kNumberOfChannels);

      // -- defaults
      for (int32_t i = 0; i < BTLElectronicsIndexing::kNumberOfChannels; ++i) {
        product->view()[i].valid() = false;
        product->view()[i].rawId() = 0;
        product->view()[i].partnerChannelId() = 0;
        product->view()[i].side() = 0;
      }

      // -- Fill from the crystals actually present in the readout map.
      int32_t nFilled = 0;
      for (const auto& detId : readoutMap.getListOfDetIds()) {
        BTLElectronicsIdPair elecIds = readoutMap.getElectronicsId(detId);

        // -- check consistent links
        const bool consistentLinks = elecIds.minus.fedId() == elecIds.plus.fedId() &&
                                     elecIds.minus.hsLinkId() == elecIds.plus.hsLinkId() &&
                                     elecIds.minus.eLinkId() == elecIds.plus.eLinkId();

        if (!consistentLinks) {
          throw cms::Exception("BTLElectronicsToDetIdMappingESProducer")
              << "crystal " << std::hex << detId.rawId() << std::dec << ": minus and plus sides on different links"
              << " (fed " << elecIds.minus.fedId() << " vs " << elecIds.plus.fedId() << ", hs "
              << elecIds.minus.hsLinkId() << " vs " << elecIds.plus.hsLinkId() << ", el " << elecIds.minus.eLinkId()
              << " vs " << elecIds.plus.eLinkId() << ")";
        }

        // -- the unpacker checks the ids read from the payload with BTLElectronicsIndexing::isValidLink():
        // a readout map that does not respect the DAQ ranges and the FED/HS-link assignment must not be used
        if (!elecIds.minus.isValid() || !elecIds.plus.isValid()) {
          throw cms::Exception("BTLElectronicsToDetIdMappingESProducer")
              << "crystal " << std::hex << detId.rawId() << std::dec << ": invalid electronics id (minus "
              << elecIds.minus << ", plus " << elecIds.plus << ")";
        }

        // -- the unpacker pairs a channel with the channel kTofhirChannelPartner[chId] of the same chip, assuming
        // that it is the other side of the same crystal (kTofhirChannelPartner is an involution, so checking
        // the plus -> minus direction is enough)
        const int32_t chPlus = elecIds.plus.channelId();
        const int32_t chMinus = elecIds.minus.channelId();
        const int32_t nChannels = static_cast<int32_t>(BTLElectronicsSpecs::kTofhirChannelPartner.size());
        if (chPlus < 0 || chPlus >= nChannels || chMinus < 0 || chMinus >= nChannels ||
            BTLElectronicsSpecs::kTofhirChannelPartner[chPlus] != chMinus) {
          throw cms::Exception("BTLElectronicsToDetIdMappingESProducer")
              << "crystal " << std::hex << detId.rawId() << std::dec << ": plus channel " << chPlus
              << " and minus channel " << chMinus << " are not partners in kTofhirChannelPartner";
        }

        const int32_t idxplus = static_cast<int32_t>(elecIds.plus.denseChannelIndex());
        const int32_t idxminus = static_cast<int32_t>(elecIds.minus.denseChannelIndex());

        // -- each electronics channel must belong to a single crystal
        for (const int32_t idx : {idxplus, idxminus}) {
          if (product->view()[idx].valid()) {
            throw cms::Exception("BTLElectronicsToDetIdMappingESProducer")
                << "crystal " << std::hex << detId.rawId() << std::dec
                << ": electronics channel (fed=" << elecIds.minus.fedId() << " hs=" << elecIds.minus.hsLinkId()
                << " el=" << elecIds.minus.eLinkId() << " ch=" << (idx == idxplus ? chPlus : chMinus)
                << ") already assigned to crystal " << std::hex << product->view()[idx].rawId() << std::dec;
          }
        }

        product->view()[idxplus].rawId() = detId.rawId();
        product->view()[idxplus].valid() = true;
        product->view()[idxplus].partnerChannelId() =
            BTLElectronicsSpecs::kTofhirChannelPartner[elecIds.plus.channelId()];
        product->view()[idxplus].side() = 1;  // plus side
        ++nFilled;

        product->view()[idxminus].rawId() = detId.rawId();
        product->view()[idxminus].valid() = true;
        product->view()[idxminus].partnerChannelId() =
            BTLElectronicsSpecs::kTofhirChannelPartner[elecIds.minus.channelId()];
        product->view()[idxminus].side() = 0;  // minus side
        ++nFilled;
      }

      LogDebug("BTLElectronicsToDetIdMappingESProducer")
          << "filled " << nFilled << " / " << BTLElectronicsIndexing::kNumberOfChannels << " table entries.";

      return product;
    }

  private:
    edm::ESGetToken<BTLReadoutMap, BTLReadoutMapRcd> readoutMapToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(BTLElectronicsToDetIdMappingESProducer);
