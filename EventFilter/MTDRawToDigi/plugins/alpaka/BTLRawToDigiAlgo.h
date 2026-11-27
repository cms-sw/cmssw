#ifndef EventFilter_MTDRawToDigi_plugins_alpaka_BTLRawToDigiAlgo_h
#define EventFilter_MTDRawToDigi_plugins_alpaka_BTLRawToDigiAlgo_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"

#include "DataFormats/FTLDigiSoA/interface/alpaka/BTLDigiDeviceCollection.h"
#include "CondFormats/MTDObjects/interface/alpaka/BTLElectronicsToDetIdMappingDevice.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsIndexing.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include <sstream>
#include <optional>

// ----------------------------------------------------------------
// Intermediate SoA produced by the decoding kernel.
// Each row corresponds to one 128-bit electronics channel payload.
// This is not yet a BTLDigi: two channel payloads belonging to the
// same chip will later be paired to build one BTLDigi entry.

// Keep local to this module rather than in DataFormats/: it's a pure auxiliary format
// (as in RecoLocalTracker/SiStripClusterizer/plugins/alpaka/SiStripRawToClusterAlgo.h)

GENERATE_SOA_LAYOUT(BTLChannelPayloadSoALayout,
                    SOA_COLUMN(uint16_t, bc0count),
                    SOA_COLUMN(bool, status),
                    SOA_COLUMN(uint32_t, bcCount),
                    SOA_COLUMN(uint16_t, t1Coarse),
                    SOA_COLUMN(uint16_t, t2Coarse),
                    SOA_COLUMN(uint16_t, eoiCoarse),
                    SOA_COLUMN(uint16_t, charge),
                    SOA_COLUMN(uint16_t, t1Fine),
                    SOA_COLUMN(uint16_t, t2Fine),
                    SOA_COLUMN(uint16_t, idleTime),
                    SOA_COLUMN(uint8_t, prevTrigF),
                    SOA_COLUMN(uint8_t, tacId),

                    // Dense index of the electronics channel,
                    // BTLElectronicsIndexing::denseChannelIndex(fedId, hsLinkId, eLinkId, chId),
                    // also used as index in the electronics -> DetId mapping.
                    // Encodes all the electronics ids of the channel:
                    //   channelDenseIndex / kChannelsPerChip = chip index (dense to group the channels of the same chip)
                    //   channelDenseIndex % kChannelsPerChip = chId
                    // -1 if (hsLinkId, eLinkId) are not valid for the FED (see BTLElectronicsIndexing::isValidLink).
                    SOA_COLUMN(int32_t, channelDenseIndex))

using BTLChannelPayloadSoA = BTLChannelPayloadSoALayout<>;
using BTLChannelPayloadHostCollection = PortableHostCollection<BTLChannelPayloadSoA>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using BTLChannelPayloadDeviceCollection = PortableCollection<BTLChannelPayloadSoA>;
}

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(BTLChannelPayloadDeviceCollection, BTLChannelPayloadHostCollection);
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Device pipeline: decode channels -> find chip segments -> pair and count digis per chip
//                  -> prefix scan of the digi offsets (cms::alpakatools::prefixScan)
//                  -> fill digis (one thread per chip, all kNumberOfChips chips).
//
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using btldigi::BTLDigiDeviceCollection;

  constexpr int kPlusSide = 1;
  constexpr int kMinusSide = 0;

  // Segment of channels of each chip, indexed by chip: [chipStart[k], chipEnd[k]).
  // Chips without data have chipStart[k] = chipEnd[k] = 0.
  struct BTLChipSegments {
    int32_t* chipStart = nullptr;  // first channel of chip k
    int32_t* chipEnd = nullptr;    // one past the last channel of chip k
  };

  class BTLRawToDigiAlgo {
  public:
    BTLChannelPayloadDeviceCollection decodeChannels(Queue& queue,
                                                     const uint64_t* rawWords_h,
                                                     const int32_t* channelFedId_h,
                                                     int32_t nChannels);

    BTLChipSegments findChipSegments(Queue& queue,
                                     BTLChannelPayloadDeviceCollection const& channelPayload_d,
                                     int32_t nChannels);

    void countDigis(Queue& queue,
                    BTLChannelPayloadDeviceCollection const& channelPayload_d,
                    BTLChipSegments const& segments,
                    BTLElectronicsToDetIdMappingDevice const& mapping,
                    int32_t nChips);

    // computes the per-segment digi offsets on device and enqueues the async copy of nDigisTot to the host
    void computeDigiOffsets(Queue& queue, int32_t nChips, cms::alpakatools::host_buffer<int32_t>& nDigisTot_h);

    BTLDigiDeviceCollection fillDigis(Queue& queue,
                                      BTLChannelPayloadDeviceCollection const& channelPayload_d,
                                      BTLElectronicsToDetIdMappingDevice const& mapping,
                                      int32_t nChips,
                                      int32_t nDigisTot);

    // ---- The processing is split in two parts, to be called from acquire() and produce() of a
    // SynchronizingEDProducer: the framework waits (non-blocking) for the queue between the two,
    // so that nDigisTot is available on the host without an explicit alpaka::wait.

    // STEP1 -> STEP2a.5: decode, find chip segments, pair and count digis, compute offsets.
    // Enqueues the async copy of the total number of digis into nDigisTot_h.
    void prepareDigis(Queue& queue,
                      const uint64_t* rawWords_h,
                      const int32_t* channelFedId_h,
                      int32_t nChannels,
                      BTLElectronicsToDetIdMappingDevice const& elecToDetId,
                      cms::alpakatools::host_buffer<int32_t>& nDigisTot_h);

    // STEP2b: allocate the output collection with nDigisTot (read on the host after the
    // synchronization) and fill it. Must be called after prepareDigis().
    BTLDigiDeviceCollection produceDigis(Queue& queue,
                                         BTLElectronicsToDetIdMappingDevice const& elecToDetId,
                                         int32_t nDigisTot);

  private:
    // ---- intermediate products, kept from prepareDigis() (acquire) to produceDigis() (produce)
    std::optional<BTLChannelPayloadDeviceCollection> channelsPayload_d_;
    BTLChipSegments chipSegments_d_;

    // ----  std::optional: buffers are (re)allocated for each event with the caching allocator, and must not be
    // reused across events: produce() does not wait for the fill kernel, so the previous event may still be using them
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> chipStart_d_;  // segment start of each chip
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> chipEnd_d_;    // segment end of each chip

    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>>
        pairPlusGlobalIndex_;  // global index in channelsPayload_d_ of the plus-side channel
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>>
        pairMinusGlobalIndex_;  // global index in channelsPayload_d_ of the minus-side channel
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>>
        digiCountPerSegment_;  // number of digis for each chip segment
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>>
        segmentDigiOffset_;  // exclusive prefix sum of digiCountPerSegment: index of the first digi of each chip
                             // in the output; the last element (index nChips) is the total number of digis
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
