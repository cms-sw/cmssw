// Occupancy of the BTL raw data, as seen by the unpacker: number of channels per event, active chips,
// channels per chip and per FED. The chip index is computed from the payload with BTLElectronicsIndexing,
// as in the unpacker (denseChannelIndex / nChannelsPerELink). No readout map needed.

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"

#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsIndexing.h"

#include "EventFilter/MTDRawToDigi/interface/BTLRawDataFormat.h"

#include "TH1F.h"
#include "TProfile.h"

class BTLRawOccupancyAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit BTLRawOccupancyAnalyzer(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<RawDataBuffer> rawDataBufferToken_;

  static constexpr std::size_t kHeaderSize = sizeof(SLinkRocketHeader_v3);
  static constexpr std::size_t kTrailerSize = sizeof(SLinkRocketTrailer_v3);
  static constexpr std::size_t kWordBytes = sizeof(btlraw::ChannelWord);  // 128 bit per channel
  static constexpr int32_t kChannelsPerChip = BTLElectronicsSpecs::nChannelsPerELink;

  // per event
  TH1F* hNChannels_;           // channels in the payload
  TH1F* hNInvalidChannels_;    // channels with (hsLinkId, eLinkId) not valid for their FED
  TH1F* hNActiveChips_;        // chips with at least one channel
  TH1F* hMaxChannelsPerChip_;  // max number of channels in a chip
  TH1F* hNFEDsWithData_;       // FEDs with at least one channel
  TH1F* hMaxChannelsPerFED_;   // max number of channels in a FED
  // per chip / per FED
  TH1F* hChannelsPerChip_;    // channels in each active chip
  TH1F* hChannelsPerFED_;     // channels in each FED with data
  TProfile* pChannelsVsFED_;  // mean number of channels vs FED id

  // summary over the job
  int64_t nEvents_ = 0;
  int64_t sumChannels_ = 0, sumActiveChips_ = 0, sumInvalid_ = 0;
  int32_t minChannels_ = std::numeric_limits<int32_t>::max(), maxChannels_ = 0;
  int32_t minActiveChips_ = std::numeric_limits<int32_t>::max(), maxActiveChips_ = 0;
  int32_t maxChannelsPerChip_ = 0, maxChannelsPerFED_ = 0;
  std::vector<int64_t> chipsWithNChannels_ = std::vector<int64_t>(kChannelsPerChip + 2, 0);  // last bin: overflow
};

BTLRawOccupancyAnalyzer::BTLRawOccupancyAnalyzer(edm::ParameterSet const& iPSet)
    : rawDataBufferToken_(consumes<RawDataBuffer>(iPSet.getParameter<edm::InputTag>("rawDataBufferTag"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  const int nFEDs = BTLElectronicsSpecs::kNumberOfFEDs;
  const int nChips = BTLElectronicsIndexing::kNumberOfChips;

  hNChannels_ = fs->make<TH1F>("hNChannels", "channels per event;channels;events", 100, 0., 60000.);
  hNInvalidChannels_ =
      fs->make<TH1F>("hNInvalidChannels", "channels with invalid link per event;channels;events", 10, -0.5, 9.5);
  hNActiveChips_ = fs->make<TH1F>("hNActiveChips", "active chips per event;active chips;events", 104, 0., nChips);
  hMaxChannelsPerChip_ = fs->make<TH1F>("hMaxChannelsPerChip",
                                        "max channels per chip in the event;max channels per chip;events",
                                        kChannelsPerChip + 2,
                                        -0.5,
                                        kChannelsPerChip + 1.5);
  hNFEDsWithData_ =
      fs->make<TH1F>("hNFEDsWithData", "FEDs with data per event;FEDs;events", nFEDs + 1, -0.5, nFEDs + 0.5);
  hMaxChannelsPerFED_ = fs->make<TH1F>(
      "hMaxChannelsPerFED", "max channels per FED in the event;max channels per FED;events", 100, 0., 2000.);
  hChannelsPerChip_ = fs->make<TH1F>("hChannelsPerChip",
                                     "channels per active chip;channels per chip;chips",
                                     kChannelsPerChip + 2,
                                     -0.5,
                                     kChannelsPerChip + 1.5);
  hChannelsPerFED_ =
      fs->make<TH1F>("hChannelsPerFED", "channels per FED with data;channels per FED;FEDs", 100, 0., 2000.);
  pChannelsVsFED_ = fs->make<TProfile>("pChannelsVsFED",
                                       "mean channels per FED;FED id;mean channels",
                                       nFEDs,
                                       BTLElectronicsSpecs::kFirstFEDId - 0.5,
                                       BTLElectronicsSpecs::kFirstFEDId + nFEDs - 0.5);
}

void BTLRawOccupancyAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  const auto& rawDataBuffer = iEvent.get(rawDataBufferToken_);

  std::vector<int32_t> channelsPerChip(BTLElectronicsIndexing::kNumberOfChips, 0);
  int32_t nChannels = 0, nInvalid = 0, nFEDsWithData = 0, maxPerFED = 0;

  for (int32_t i = 0; i < BTLElectronicsSpecs::kNumberOfFEDs; ++i) {
    const int32_t fed = BTLElectronicsSpecs::kFirstFEDId + i;
    const auto& fragData = rawDataBuffer.fragmentData(fed);

    int32_t nChInFed = 0;
    if (fragData.size() >= kHeaderSize + kTrailerSize) {
      const auto payloadSpan = fragData.payload(kHeaderSize, kTrailerSize);
      nChInFed = static_cast<int32_t>(payloadSpan.size() / kWordBytes);

      for (int32_t iWord = 0; iWord < nChInFed; ++iWord) {
        // same decoding as DumpBTLRawDataBuffer
        btlraw::ChannelWord w;
        std::memcpy(&w, payloadSpan.data() + iWord * kWordBytes, kWordBytes);
        const int32_t hslink = static_cast<int32_t>(btlraw::extractBits(w, btlraw::kHSLink));
        const int32_t elink = static_cast<int32_t>(btlraw::extractBits(w, btlraw::kELink));
        const int32_t chId = static_cast<int32_t>(btlraw::extractBits(w, btlraw::kChId));

        if (!BTLElectronicsIndexing::isValidLink(fed, hslink, elink)) {
          ++nInvalid;
          continue;
        }
        ++channelsPerChip[BTLElectronicsIndexing::denseChannelIndex(fed, hslink, elink, chId) / kChannelsPerChip];
      }
    } else if (fragData.size() > 0) {
      edm::LogWarning("BTLRawOccupancyAnalyzer")
          << "FED " << fed << ": fragment of " << fragData.size() << " bytes, smaller than header + trailer";
    }

    pChannelsVsFED_->Fill(fed, nChInFed);
    if (nChInFed > 0) {
      ++nFEDsWithData;
      hChannelsPerFED_->Fill(nChInFed);
    }
    nChannels += nChInFed;
    maxPerFED = std::max(maxPerFED, nChInFed);
  }

  int32_t nActiveChips = 0, maxPerChip = 0;
  for (const int32_t n : channelsPerChip) {
    if (n == 0)
      continue;
    ++nActiveChips;
    maxPerChip = std::max(maxPerChip, n);
    hChannelsPerChip_->Fill(n);
    ++chipsWithNChannels_[std::min(n, kChannelsPerChip + 1)];
  }

  hNChannels_->Fill(nChannels);
  hNInvalidChannels_->Fill(nInvalid);
  hNActiveChips_->Fill(nActiveChips);
  hMaxChannelsPerChip_->Fill(maxPerChip);
  hNFEDsWithData_->Fill(nFEDsWithData);
  hMaxChannelsPerFED_->Fill(maxPerFED);

  edm::LogInfo("BTLRawOccupancyAnalyzer")
      << "run " << iEvent.id().run() << " lumi " << iEvent.id().luminosityBlock() << " event " << iEvent.id().event()
      << ": channels " << nChannels << " (invalid link " << nInvalid << "), active chips " << nActiveChips
      << ", max channels per chip " << maxPerChip << ", FEDs with data " << nFEDsWithData << ", max channels per FED "
      << maxPerFED;

  ++nEvents_;
  sumChannels_ += nChannels;
  sumActiveChips_ += nActiveChips;
  sumInvalid_ += nInvalid;
  minChannels_ = std::min(minChannels_, nChannels);
  maxChannels_ = std::max(maxChannels_, nChannels);
  minActiveChips_ = std::min(minActiveChips_, nActiveChips);
  maxActiveChips_ = std::max(maxActiveChips_, nActiveChips);
  maxChannelsPerChip_ = std::max(maxChannelsPerChip_, maxPerChip);
  maxChannelsPerFED_ = std::max(maxChannelsPerFED_, maxPerFED);
}

void BTLRawOccupancyAnalyzer::endJob() {
  if (nEvents_ == 0) {
    edm::LogPrint("BTLRawOccupancyAnalyzer") << "BTLRawOccupancyAnalyzer: no events";
    return;
  }

  int64_t nChipsTot = 0;
  for (const auto n : chipsWithNChannels_)
    nChipsTot += n;

  std::ostringstream os;
  os << "========== BTLRawOccupancyAnalyzer summary (" << nEvents_ << " events) ==========\n"
     << "channels per event     : mean " << double(sumChannels_) / nEvents_ << ", min " << minChannels_ << ", max "
     << maxChannels_ << "\n"
     << "invalid-link channels  : total " << sumInvalid_ << "\n"
     << "active chips per event : mean " << double(sumActiveChips_) / nEvents_ << ", min " << minActiveChips_
     << ", max " << maxActiveChips_ << " (out of " << BTLElectronicsIndexing::kNumberOfChips << ")\n"
     << "channels per active chip: mean " << (nChipsTot > 0 ? double(sumChannels_ - sumInvalid_) / nChipsTot : 0.)
     << ", max " << maxChannelsPerChip_ << "\n"
     << "max channels per FED   : " << maxChannelsPerFED_ << "\n"
     << "active chips with N channels (all events):\n";
  for (int32_t n = 1; n <= kChannelsPerChip + 1; ++n) {
    if (chipsWithNChannels_[n] == 0)
      continue;
    os << "  N = " << (n <= kChannelsPerChip ? std::to_string(n) : std::string("> 32")) << " : "
       << chipsWithNChannels_[n] << " (" << 100. * chipsWithNChannels_[n] / nChipsTot << " %)\n";
  }
  edm::LogPrint("BTLRawOccupancyAnalyzer") << os.str();
}

void BTLRawOccupancyAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawDataBufferTag", edm::InputTag("btlDigiToRaw"));
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(BTLRawOccupancyAnalyzer);
