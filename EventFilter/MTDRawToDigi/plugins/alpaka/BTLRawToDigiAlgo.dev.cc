#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "EventFilter/MTDRawToDigi/interface/BTLRawDataFormat.h"

#include "BTLRawToDigiAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // number of channels of a chip (e-link)
  constexpr int32_t kChannelsPerChip = BTLElectronicsSpecs::nChannelsPerELink;

  //---------------------------------------------------------------------
  // KERNEL 1: one thread per raw channel payload.
  // Fills the device-side auxiliary SoA
  //---------------------------------------------------------------------
  class BTLDecodeChannelsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint64_t* rawWords,     // 2 x uint64 per channel
                                  const int32_t* channelFedId,  // one entry per channel, filled host-side
                                  int32_t nChannels,
                                  BTLChannelPayloadSoA::View out) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nChannels)) {
        // the bit layout is defined in BTLRawDataFormat.h
        using namespace btlraw;
        ChannelWord w;
        w.lo = rawWords[2 * i];
        w.hi = rawWords[2 * i + 1];
        out[i].bc0count() = static_cast<uint16_t>(extractBits(w, kBC0count.pos, kBC0count.width));
        out[i].status() = extractBits(w, kStatus.pos, kStatus.width) != 0;
        const int32_t hslink = static_cast<int32_t>(extractBits(w, kHSLink.pos, kHSLink.width));
        const int32_t elink = static_cast<int32_t>(extractBits(w, kELink.pos, kELink.width));
        const int32_t chId = static_cast<int32_t>(extractBits(w, kChId.pos, kChId.width));
        out[i].bcCount() = static_cast<uint32_t>(extractBits(w, kBCcount.pos, kBCcount.width));
        out[i].t1Coarse() = static_cast<uint16_t>(extractBits(w, kT1Coarse.pos, kT1Coarse.width));
        out[i].t2Coarse() = static_cast<uint16_t>(extractBits(w, kT2Coarse.pos, kT2Coarse.width));
        out[i].eoiCoarse() = static_cast<uint16_t>(extractBits(w, kEOICoarse.pos, kEOICoarse.width));
        out[i].charge() = static_cast<uint16_t>(extractBits(w, kCharge.pos, kCharge.width));
        out[i].t1Fine() = static_cast<uint16_t>(extractBits(w, kT1Fine.pos, kT1Fine.width));
        out[i].t2Fine() = static_cast<uint16_t>(extractBits(w, kT2Fine.pos, kT2Fine.width));
        out[i].idleTime() = static_cast<uint16_t>(extractBits(w, kIdleTime.pos, kIdleTime.width));
        out[i].prevTrigF() = static_cast<uint8_t>(extractBits(w, kPrevTrigF.pos, kPrevTrigF.width));
        out[i].tacId() = static_cast<uint8_t>(extractBits(w, kTACId.pos, kTACId.width));

        // Dense index of the electronics channel (also the index in the mapping):
        // the chip index (dense, in [0, nChips)) is channelDenseIndex / kChannelsPerChip.
        // Channels with (hsLinkId, eLinkId) not valid for their FED get channelDenseIndex = -1 and are not
        // assigned to any chip.
        // The check is needed since in the payload hsLinkId has 6 bits (0-63) and eLinkId 5 bits (0-31):
        // corrupted values outside the BTL ranges (36 HS-links per Serenity, 24 e-links per HS-link), or an
        // HS-link not belonging to the FED, would give the index of another chip or an index outside the mapping.
        // The FED id is not checked: it comes from the FED loop on the host.
        // Output row i = input channel i: the input order is kept (used by BTLFindChipSegmentsKernel).
        const int32_t fed = channelFedId[i];
        out[i].channelDenseIndex() = BTLElectronicsIndexing::isValidLink(fed, hslink, elink)
                                         ? BTLElectronicsIndexing::denseChannelIndex(fed, hslink, elink, chId)
                                         : -1;
      }
    }
  };

  //---------------------------------------------------------------------
  // KERNEL 1.5: finds the segment of channels corresponding to each chip, one thread per channel.
  // In the BTLChannelPayloadSoA filled in STEP1 the channels of the same chip are contiguous, and the chip
  // key (channelDenseIndex / kChannelsPerChip) is a dense index in [0, nChips): the segment of chip k is stored
  // directly at index k,
  //   chipStart[k] = first channel of chip k, chipEnd[k] = (last channel of chip k) + 1
  // Each thread checks if its channel is the first and/or the last one of its chip, so that each
  // entry is written by exactly one thread.
  // Chips without data keep chipStart = chipEnd = 0 (memset before the kernel), i.e. an empty segment.
  // Channels with channelDenseIndex = -1 (corrupted hsLinkId/eLinkId) are not assigned to any chip.
  //-----------------------------------------------------------------------------------------------------
  class BTLFindChipSegmentsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  BTLChannelPayloadSoA::ConstView channelData,
                                  int32_t nChannels,
                                  int32_t* chipStart,
                                  int32_t* chipEnd) const {
      constexpr int32_t nCh = kChannelsPerChip;
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nChannels)) {
        const int32_t idx = channelData[i].channelDenseIndex();
        if (idx < 0)
          continue;
        const int32_t key = idx / nCh;
        // chip of the previous and next channels: -1 if the neighbour does not exist or has channelDenseIndex = -1
        // (checked explicitly, since -1 / nCh = 0 is the index of chip 0)
        const int32_t prev = (i > 0) ? channelData[i - 1].channelDenseIndex() : -1;
        const int32_t next = (i < nChannels - 1) ? channelData[i + 1].channelDenseIndex() : -1;
        const int32_t prevKey = (prev < 0) ? -1 : prev / nCh;
        const int32_t nextKey = (next < 0) ? -1 : next / nCh;
        if (prevKey != key)
          chipStart[key] = i;
        if (nextKey != key)
          chipEnd[key] = i + 1;
      }
    }
  };

  //---------------------------------------------------------------------
  // KERNEL 2a: one thread per chip: pair plus/minus channels and count digis per chip segment.
  // For each channel, looks for its partner channel in the same chip; complete pairs are stored
  // once (from the plus side).
  // The segment index is the chip index (channelDenseIndex / kChannelsPerChip): chips without
  // data have an empty segment, and digiCountPerSegment is written for all the chips (0 for chips without data).
  // Output arrays (size nChips * nPairsPerChip):
  //   pairPlusGlobalIndex[seg * nPairsPerChip + k]  = index in channelPayload_d of the plus-side channel
  //                                                   of the k-th digi in segment seg (-1 if no plus-side hit)
  //   pairMinusGlobalIndex[seg * nPairsPerChip + k] = same for the minus-side channel (-1 if absent)
  //   digiCountPerSegment[seg]                      = number of digis found in segment seg
  //---------------------------------------------------------------------
  class BTLPairAndCountDigisKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  BTLChannelPayloadSoA::ConstView channelData,
                                  const int32_t* chipStart,
                                  const int32_t* chipEnd,
                                  BTLElectronicsToDetIdMappingSoA::ConstView mapping,
                                  int32_t nChips,
                                  int32_t* pairPlusGlobalIndex,  // size nChips * nPairsPerChip
                                  int32_t* pairMinusGlobalIndex,
                                  int32_t* digiCountPerSegment) const {
      for (int32_t segment : cms::alpakatools::uniform_elements(acc, nChips)) {
        const int32_t start = chipStart[segment];
        const int32_t end = chipEnd[segment];

        const int32_t base = segment * BTLElectronicsIndexing::nPairsPerChip;

        int32_t nDigis = 0;  // local thread counter

        for (int32_t i = start; i < end; ++i) {
          // a chip has at most nPairsPerChip digis (one per crystal): more digis are possible only with
          // corrupted data (e.g. a duplicated chId), and would be written in the slots of the next chip.
          // The remaining channels of the chip are discarded.
          if (nDigis >= BTLElectronicsIndexing::nPairsPerChip)
            break;

          // all the channels of a segment have a valid channelDenseIndex (>= 0), see BTLFindChipSegmentsKernel
          const int32_t denseIndex = channelData[i].channelDenseIndex();

          const auto& m = mapping[denseIndex];

          if (!m.valid()) {
#ifdef EDM_ML_DEBUG
            printf("INVALID MAPPING: chip=%d ch=%d denseIndex=%d\n",
                   denseIndex / kChannelsPerChip,
                   denseIndex % kChannelsPerChip,
                   denseIndex);
#endif
            continue;
          }

          // Find the partner channel within this chip: same chip, channel partnerChannelId
          const int32_t partnerDenseIndex = (denseIndex / kChannelsPerChip) * kChannelsPerChip + m.partnerChannelId();
          int32_t partnerRow = -1;
          for (int32_t j = start; j < end; ++j) {
            if (j == i)
              continue;
            if (channelData[j].channelDenseIndex() == partnerDenseIndex) {
              partnerRow = j;
              break;
            }
          }

          if (partnerRow >= 0) {
            // Complete pair: count it only from the plus side.
            if (m.side() == kPlusSide) {
              pairPlusGlobalIndex[base + nDigis] = i;  // i = global index of this channel
              pairMinusGlobalIndex[base + nDigis] = partnerRow;
              ++nDigis;
            }
          } else {
            // Single-side digi.
            if (m.side() == kPlusSide) {
              pairPlusGlobalIndex[base + nDigis] = i;
              pairMinusGlobalIndex[base + nDigis] = -1;
            } else {
              pairPlusGlobalIndex[base + nDigis] = -1;
              pairMinusGlobalIndex[base + nDigis] = i;
            }
            ++nDigis;
          }
        }

        digiCountPerSegment[segment] = nDigis;
      }
    }
  };

  //---------------------------------------------------------------------
  // KERNEL 2b: fill output BTLDigiSoA, one thread per chip segment
  // Reads the (plus, minus) index pairs built by BTLPairAndCountDigisKernel and
  // the per-segment digi offsets to write each digi at the correct position in
  // the output collection. Digis are ordered by chip index and the digis of the same chip are contiguous.
  // No channel pairing is redone here.
  //---------------------------------------------------------------------
  class BTLFillDigisKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  BTLChannelPayloadSoA::ConstView channelData,
                                  const int32_t* digiCountPerSegment,
                                  const int32_t* segmentDigiOffset,
                                  const int32_t* pairPlusGlobalIndex,
                                  const int32_t* pairMinusGlobalIndex,
                                  BTLElectronicsToDetIdMappingSoA::ConstView mapping,
                                  int32_t nChips,
                                  ::btldigi::BTLDigiSoA::View digisOut) const {
      // chips without data have digiCountPerSegment = 0 and write nothing
      for (int32_t segment : cms::alpakatools::uniform_elements(acc, nChips)) {
        const int32_t base = segment * BTLElectronicsIndexing::nPairsPerChip;
        const int32_t writeBase = segmentDigiOffset[segment];

        for (int32_t k = 0; k < digiCountPerSegment[segment]; ++k) {
          const int32_t idxPlus = pairPlusGlobalIndex[base + k];
          const int32_t idxMinus = pairMinusGlobalIndex[base + k];
          const int32_t validIdx = (idxMinus >= 0) ? idxMinus : idxPlus;

          const auto& m = mapping[channelData[validIdx].channelDenseIndex()];

          auto d = digisOut[writeBase + k];
          d.rawId() = m.rawId();

          const auto& cValid = channelData[validIdx];
          d.BC0count() = cValid.bc0count();
          d.status() = cValid.status();
          d.BCcount() = cValid.bcCount();

          // minus side - set to default values (0 for now) if not present
          if (idxMinus >= 0) {
            const auto& cm = channelData[idxMinus];
            d.chIDMinus() = cm.channelDenseIndex() % kChannelsPerChip;
            d.T1coarseMinus() = cm.t1Coarse();
            d.T2coarseMinus() = cm.t2Coarse();
            d.EOIcoarseMinus() = cm.eoiCoarse();
            d.ChargeMinus() = cm.charge();
            d.T1fineMinus() = cm.t1Fine();
            d.T2fineMinus() = cm.t2Fine();
            d.IdleTimeMinus() = cm.idleTime();
            d.PrevTrigFMinus() = cm.prevTrigF();
            d.TACIDMinus() = cm.tacId();
          } else {
            d.chIDMinus() = 0;
            d.T1coarseMinus() = 0;
            d.T2coarseMinus() = 0;
            d.EOIcoarseMinus() = 0;
            d.ChargeMinus() = 0;
            d.T1fineMinus() = 0;
            d.T2fineMinus() = 0;
            d.IdleTimeMinus() = 0;
            d.PrevTrigFMinus() = 0;
            d.TACIDMinus() = 0;
          }
          // plus side - set to default values (0 for now) if not present
          if (idxPlus >= 0) {
            const auto& cp = channelData[idxPlus];
            d.chIDPlus() = cp.channelDenseIndex() % kChannelsPerChip;
            d.T1coarsePlus() = cp.t1Coarse();
            d.T2coarsePlus() = cp.t2Coarse();
            d.EOIcoarsePlus() = cp.eoiCoarse();
            d.ChargePlus() = cp.charge();
            d.T1finePlus() = cp.t1Fine();
            d.T2finePlus() = cp.t2Fine();
            d.IdleTimePlus() = cp.idleTime();
            d.PrevTrigFPlus() = cp.prevTrigF();
            d.TACIDPlus() = cp.tacId();
          } else {
            d.chIDPlus() = 0;
            d.T1coarsePlus() = 0;
            d.T2coarsePlus() = 0;
            d.EOIcoarsePlus() = 0;
            d.ChargePlus() = 0;
            d.T1finePlus() = 0;
            d.T2finePlus() = 0;
            d.IdleTimePlus() = 0;
            d.PrevTrigFPlus() = 0;
            d.TACIDPlus() = 0;
          }
        }
      }
    }
  };

  //---------------------------------------------------------------------
  // BTLRawToDigiAlgo
  //---------------------------------------------------------------------
  BTLChannelPayloadDeviceCollection BTLRawToDigiAlgo::decodeChannels(Queue& queue,
                                                                     const uint64_t* rawWords_h,
                                                                     const int32_t* channelFedId_h,
                                                                     int32_t nChannels) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi [decodeChannels] START ========== \n"
                             << " decoding " << nChannels << " channels";
#endif

    // Copy host --> device
    auto rawWords_d = cms::alpakatools::make_device_buffer<uint64_t[]>(queue, 2 * nChannels);
    auto channelFedId_d = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChannels);
    alpaka::memcpy(queue, rawWords_d, cms::alpakatools::make_host_view(rawWords_h, 2 * nChannels));
    alpaka::memcpy(queue, channelFedId_d, cms::alpakatools::make_host_view(channelFedId_h, nChannels));

    BTLChannelPayloadDeviceCollection channelsPayload_d(queue, nChannels);

    auto workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(uint32_t(nChannels), 256u), 256u);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        BTLDecodeChannelsKernel{},
                        rawWords_d.data(),
                        channelFedId_d.data(),
                        nChannels,
                        channelsPayload_d.view());

    return channelsPayload_d;
  }

  BTLChipSegments BTLRawToDigiAlgo::findChipSegments(Queue& queue,
                                                     BTLChannelPayloadDeviceCollection const& channelPayload_d,
                                                     int32_t nChannels) {
    const int32_t nChips = BTLElectronicsIndexing::kNumberOfChips;

#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi [findChipSegments] START ==========";
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi [findChipSegments] : total number of chips = " << nChips;
#endif

    // allocated for each event (cheap with the caching allocator): the previous event may still be
    // using its buffers on another queue, since produce() does not wait for the fill kernel
    chipStart_d_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips));
    chipEnd_d_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips));

    // chips without data must have an empty segment (start = end = 0)
    alpaka::memset(queue, *chipStart_d_, 0);
    alpaka::memset(queue, *chipEnd_d_, 0);

    // one thread per channel
    auto workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(uint32_t(nChannels), 256u), 256u);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        BTLFindChipSegmentsKernel{},
                        channelPayload_d.const_view(),
                        nChannels,
                        chipStart_d_->data(),
                        chipEnd_d_->data());

#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi [findChipSegments] : kernel launched, returning "
                                "chipStart_d_, chipEnd_d_ ==========";
#endif
    return {chipStart_d_->data(), chipEnd_d_->data()};
  }

  void BTLRawToDigiAlgo::countDigis(Queue& queue,
                                    BTLChannelPayloadDeviceCollection const& channelPayload_d,
                                    BTLChipSegments const& segments,
                                    BTLElectronicsToDetIdMappingDevice const& mapping,
                                    int32_t nChips) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi countDigis() START ==========";
#endif

    // -- buffers dimensioned on the total number of chips, allocated for each event (see findChipSegments)
    digiCountPerSegment_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips));
    pairPlusGlobalIndex_.emplace(
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips * BTLElectronicsIndexing::nPairsPerChip));
    pairMinusGlobalIndex_.emplace(
        cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips * BTLElectronicsIndexing::nPairsPerChip));
    segmentDigiOffset_.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nChips + 1));

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(uint32_t(nChips), 256u), 256u);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        BTLPairAndCountDigisKernel{},
                        channelPayload_d.const_view(),
                        segments.chipStart,
                        segments.chipEnd,
                        mapping.const_view(),
                        nChips,
                        pairPlusGlobalIndex_->data(),
                        pairMinusGlobalIndex_->data(),
                        digiCountPerSegment_->data());

#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi [countDigis] : kernel launched ==========";
#endif
  }

  void BTLRawToDigiAlgo::computeDigiOffsets(Queue& queue,
                                            int32_t nChips,
                                            cms::alpakatools::host_buffer<int32_t>& nDigisTot_h) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi computeDigiOffsets() START ==========";
#endif

    // STEP 2a.5: digi offsets per chip segment, as an exclusive prefix sum of digiCountPerSegment:
    //   segmentDigiOffset[s]      = index of the first digi of segment s in the output BTLDigiSoA
    //   segmentDigiOffset[nChips] = nDigisTot = total number of digis in the event
    // The inclusive prefix scan (cms::alpakatools::prefixScan) is written shifted by one element,
    // into segmentDigiOffset + 1, so that segmentDigiOffset[0] = 0 and the result is the exclusive scan.
    // NOTE: the scan runs over all the nChips; digiCountPerSegment is 0 for the chips without data.
    alpaka::memset(queue, *segmentDigiOffset_, 0);

    cms::alpakatools::prefixScan<Acc1D>(
        queue, digiCountPerSegment_->data(), segmentDigiOffset_->data() + 1, uint32_t(nChips));

    // only device -> host copy: nDigisTot (4 bytes, last element of segmentDigiOffset), needed to allocate
    // the output collection.
    // No wait here: the value is read in produce(), after the framework has synchronized the queue.
    alpaka::memcpy(queue, nDigisTot_h, cms::alpakatools::make_device_view(queue, segmentDigiOffset_->data()[nChips]));
  }

  BTLDigiDeviceCollection BTLRawToDigiAlgo::fillDigis(Queue& queue,
                                                      BTLChannelPayloadDeviceCollection const& channelPayload_d,
                                                      BTLElectronicsToDetIdMappingDevice const& mapping,
                                                      int32_t nChips,
                                                      int32_t nDigisTot) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi fillDigis() START ==========";
    LogDebug("BTLRawToDigi") << " === BTLRawToDigi fillDigis(): total number of digis = " << nDigisTot;
#endif

    // protection against nDigisTot = 0 (e.g. wrong mapping or all channels corrupted/unmapped)
    if (nDigisTot == 0)
      return BTLDigiDeviceCollection(queue, 0);

    BTLDigiDeviceCollection digis_d(queue, nDigisTot);

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(uint32_t(nChips), 256u), 256u);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        BTLFillDigisKernel{},
                        channelPayload_d.const_view(),
                        digiCountPerSegment_->data(),
                        segmentDigiOffset_->data(),
                        pairPlusGlobalIndex_->data(),
                        pairMinusGlobalIndex_->data(),
                        mapping.const_view(),
                        nChips,
                        digis_d.view());

    return digis_d;
  }

  void BTLRawToDigiAlgo::prepareDigis(Queue& queue,
                                      const uint64_t* rawWords_h,
                                      const int32_t* channelFedId_h,
                                      int32_t nChannels,
                                      BTLElectronicsToDetIdMappingDevice const& elecToDetIdMapping,
                                      cms::alpakatools::host_buffer<int32_t>& nDigisTot_h) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi prepareDigis() START ==========";
#endif

    // -- STEP1: Launch decode for each channel
    channelsPayload_d_.emplace(
        decodeChannels(queue, rawWords_h, channelFedId_h, nChannels));  // auxiliary soa on device
    auto const& channelsPayload_d = *channelsPayload_d_;

    // -- DEBUG STEP1: copy to host and print
#ifdef EDM_ML_DEBUG
    BTLChannelPayloadHostCollection channelsPayload_h(std::max(nChannels, 1));
    alpaka::memcpy(queue, channelsPayload_h.buffer(), channelsPayload_d.buffer());
    alpaka::wait(queue);
    auto soa = channelsPayload_h.view();

    std::ostringstream os;
    for (int i = 0; i < nChannels; ++i) {
      const int32_t idx = soa.channelDenseIndex()[i];
      os << "i = " << i << " channelDenseIndex = " << idx;
      if (idx >= 0)
        os << " chip = " << idx / kChannelsPerChip << " chId = " << idx % kChannelsPerChip;
      os << "\n";
    }
    LogDebug("BTLRawToDigi") << os.str();
#endif

    // -- STEP1.5: find segments corresponding to each active chip
    chipSegments_d_ = findChipSegments(queue, channelsPayload_d, nChannels);

    const int32_t nChips = BTLElectronicsIndexing::kNumberOfChips;

    // -- DEBUG findChipSegments
#ifdef EDM_ML_DEBUG
    auto chipStart_h = cms::alpakatools::make_host_buffer<int32_t[]>(queue, nChips);
    auto chipEnd_h = cms::alpakatools::make_host_buffer<int32_t[]>(queue, nChips);
    alpaka::memcpy(queue, chipStart_h, *chipStart_d_);
    alpaka::memcpy(queue, chipEnd_h, *chipEnd_d_);
    alpaka::wait(queue);

    int32_t nActiveChips = 0;
    std::ostringstream os1;
    for (int32_t k = 0; k < nChips; ++k) {  // print only the chips with data
      if (chipEnd_h[k] > chipStart_h[k]) {
        ++nActiveChips;
        os1 << "chip " << k << " : [" << chipStart_h[k] << ", " << chipEnd_h[k] << ")"
            << " size=" << chipEnd_h[k] - chipStart_h[k] << '\n';
      }
    }
    LogDebug("BTLRawToDigi") << "Debugging findChipSegments...";
    LogDebug("BTLRawToDigi") << "Number of active chips = " << nActiveChips;
    LogDebug("BTLRawToDigi") << os1.str();
#endif

    // STEP 2a: pairing and count digis, one thread per chip (nChips = maximum number of chips);
    // chips without data have an empty segment and give 0 digis.
    countDigis(queue, channelsPayload_d, chipSegments_d_, elecToDetIdMapping, nChips);

    // -- DEBUG pair and count digis
#ifdef EDM_ML_DEBUG
    auto digiCount_h =
        cms::alpakatools::make_host_buffer<int32_t[]>(queue, alpaka::getExtentProduct(*digiCountPerSegment_));
    alpaka::memcpy(queue, digiCount_h, *digiCountPerSegment_);
    alpaka::wait(queue);

    std::ostringstream os2;
    for (int32_t k = 0; k < nChips; ++k) {
      if (chipEnd_h[k] > chipStart_h[k])
        os2 << "chip " << k << "  number of digis in this chip : " << digiCount_h[k] << '\n';
    }
    LogDebug("BTLRawToDigi") << os2.str();
#endif

    // -- STEP2a.5: digi offsets per segment and total number of digis (async copy to nDigisTot_h)
    computeDigiOffsets(queue, nChips, nDigisTot_h);
  }

  BTLDigiDeviceCollection BTLRawToDigiAlgo::produceDigis(Queue& queue,
                                                         BTLElectronicsToDetIdMappingDevice const& elecToDetIdMapping,
                                                         int32_t nDigisTot) {
#ifdef EDM_ML_DEBUG
    LogDebug("BTLRawToDigi") << "========== BTLRawToDigi produceDigis() START ==========";
#endif

    const int32_t nChips = BTLElectronicsIndexing::kNumberOfChips;

    // -- STEP2b: fill digis
    auto digis_d = fillDigis(queue, *channelsPayload_d_, elecToDetIdMapping, nChips, nDigisTot);

    // the intermediate SoA is not needed anymore: release it (the caching allocator frees it
    // only once the work already enqueued on the queue has completed)
    channelsPayload_d_.reset();

    // -- DEBUG FILL DIGIS
#ifdef EDM_ML_DEBUG
    ::btldigi::BTLDigiHostCollection digis_h(queue, std::max(nDigisTot, 1));
    alpaka::memcpy(queue, digis_h.buffer(), digis_d.buffer());
    alpaka::wait(queue);
    auto digis_h_view = digis_h.view();
    std::ostringstream os3;
    for (int32_t i = 0; i < nDigisTot; ++i) {
      os3 << "DIGI " << i << " rawId=" << digis_h_view[i].rawId() << " BC0=" << digis_h_view[i].BC0count()
          << " status=" << digis_h_view[i].status() << " BCcount=" << digis_h_view[i].BCcount()
          << " plus(ch=" << int(digis_h_view[i].chIDPlus()) << ", charge=" << digis_h_view[i].ChargePlus() << ")"
          << " minus(ch=" << int(digis_h_view[i].chIDMinus()) << ", charge=" << digis_h_view[i].ChargeMinus() << ")"
          << '\n';
    }
    LogDebug("BTLRawToDigi") << os3.str();
#endif

    // -- return the filled digi collection
    return digis_d;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
