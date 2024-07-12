#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "EventFilter/EcalRawToDigi/interface/ElectronicsIdGPU.h"
#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "UnpackPortable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::raw {

  using namespace ::ecal::raw;
  using namespace cms::alpakatools;

  class Kernel_unpack {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  unsigned char const* __restrict__ data,
                                  uint32_t const* __restrict__ offsets,
                                  int const* __restrict__ feds,
                                  EcalDigiDeviceCollection::View digisDevEB,
                                  EcalDigiDeviceCollection::View digisDevEE,
                                  EcalElectronicsMappingDevice::ConstView eid2did,
                                  uint32_t const nbytesTotal) const {
      constexpr auto kSampleSize = ecalPh1::sampleSize;

      // indices
      auto const ifed = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      auto const threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];

      // offset in bytes
      auto const offset = offsets[ifed];
      // fed id
      auto const fed = feds[ifed];
      auto const isBarrel = is_barrel(static_cast<uint8_t>(fed - 600));
      // size
      auto const gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      auto const size = ifed == gridDim - 1 ? nbytesTotal - offset : offsets[ifed + 1] - offset;
      auto* samples = isBarrel ? digisDevEB.data()->data() : digisDevEE.data()->data();
      auto* ids = isBarrel ? digisDevEB.id() : digisDevEE.id();
      auto* pChannelsCounter = isBarrel ? &digisDevEB.size() : &digisDevEE.size();

      // offset to the right raw buffer
      uint64_t const* buffer = reinterpret_cast<uint64_t const*>(data + offset);

      // dump first 3 bits for each 64-bit word
      //print_first3bits(buffer, size / 8);

      //
      // fed header
      //
      auto const fed_header = buffer[0];
      uint32_t bx = (fed_header >> H_BX_B) & H_BX_MASK;
      uint32_t lv1 = (fed_header >> H_L1_B) & H_L1_MASK;
      uint32_t triggerType = (fed_header >> H_TTYPE_B) & H_TTYPE_MASK;

      // determine the number of FE channels from the trigger type
      uint32_t numbChannels(0);
      if (triggerType == PHYSICTRIGGER) {
        numbChannels = NUMB_FE;
      } else if (triggerType == CALIBRATIONTRIGGER) {
        numbChannels = NUMB_FE + 2;  // FE + 2 MEM blocks
      } else {
        // unsupported trigger type
        return;
      }

      // 9 for fed + dcc header
      // 36 for 4 EE TCC blocks or 18 for 1 EB TCC block
      // 6 for SR block size

      // dcc header w2
      auto const w2 = buffer[2];
      uint8_t const fov = (w2 >> H_FOV_B) & H_FOV_MASK;

      // make a list of channels with data from DCC header channels status
      // this could be done for each block instead of each thread since it defined per FED
      uint8_t exp_ttids[NUMB_FE + 2];  // FE + 2 MEM blocks
      uint8_t ch = 1;
      uint8_t nCh = 0;
      for (uint8_t i = 4; i < 9; ++i) {           // data words with channel status info
        for (uint8_t j = 0; j < 14; ++j, ++ch) {  // channel status fields in one data word
          const uint8_t shift = j * 4;            //each channel has 4 bits
          const int chStatus = (buffer[i] >> shift) & H_CHSTATUS_MASK;
          const bool regular = (chStatus == CH_DISABLED || chStatus == CH_SUPPRESS);
          const bool problematic =
              (chStatus == CH_TIMEOUT || chStatus == CH_HEADERERR || chStatus == CH_LINKERR ||
               chStatus == CH_LENGTHERR || chStatus == CH_IFIFOFULL || chStatus == CH_L1AIFIFOFULL);
          if (!(regular || problematic)) {
            exp_ttids[nCh] = ch;
            ++nCh;
          }
        }
      }

      //
      // print Tower block headers
      //
      uint8_t ntccblockwords = isBarrel ? 18 : 36;
      auto const* tower_blocks_start = buffer + 9 + ntccblockwords + 6;
      auto const* trailer = buffer + (size / 8 - 1);
      auto const* current_tower_block = tower_blocks_start;
      uint8_t iCh = 0;
      uint8_t next_tower_id = exp_ttids[iCh];
      while (current_tower_block < trailer && iCh < numbChannels) {
        auto const w = *current_tower_block;
        uint8_t ttid = w & TOWER_ID_MASK;
        uint16_t bxlocal = (w >> TOWER_BX_B) & TOWER_BX_MASK;
        uint16_t lv1local = (w >> TOWER_L1_B) & TOWER_L1_MASK;
        uint16_t block_length = (w >> TOWER_LENGTH_B) & TOWER_LENGTH_MASK;

        // fast forward to the next good tower id (in case of recovery from an earlier header corruption)
        while (exp_ttids[iCh] < next_tower_id) {
          ++iCh;
        }
        ++iCh;

        // check if the tower id in the tower header is the one expected
        // if not try to find the next good header, point the current_tower_block to it, and extract its tower id
        // or break if there is none
        if (ttid != next_tower_id) {
          next_tower_id = find_next_tower_block(current_tower_block, trailer, bx, lv1);
          if (next_tower_id < TOWER_ID_MASK) {
            continue;
          } else {
            break;
          }
        }

        // prepare for the next iteration
        next_tower_id = exp_ttids[iCh];

        uint16_t const dccbx = bx & 0xfff;
        uint16_t const dccl1 = lv1 & 0xfff;
        // fov>=1 is required to support simulated data for which bx==bxlocal==0
        if (fov >= 1 && !is_synced_towerblock(dccbx, bxlocal, dccl1, lv1local)) {
          current_tower_block += block_length;
          continue;
        }

        // go through all the channels
        // get the next channel coordinates
        uint32_t const nchannels = (block_length - 1) / 3;

        bool bad_block = false;
        auto& ch_with_bad_block = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
        if (once_per_block(acc)) {
          ch_with_bad_block = std::numeric_limits<uint32_t>::max();
        }
        // make sure the shared memory is initialised for all threads
        alpaka::syncBlockThreads(acc);

        auto const threadsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        // 1 threads per channel in this block
        // All threads enter the loop regardless if they will treat channel indices channel >= nchannels.
        // The threads with excess indices perform no operations but also reach the syncBlockThreads() inside the loop.
        for (uint32_t i = 0; i < nchannels; i += threadsPerBlock) {
          auto const channel = i + threadIdx;

          uint64_t wdata;
          uint8_t stripid;
          uint8_t xtalid;

          // threads must be inside the range (no break here because of syncBlockThreads() afterwards)
          if (channel < nchannels && channel < ch_with_bad_block) {
            // inc the channel's counter and get the pos where to store
            wdata = current_tower_block[1 + channel * 3];
            stripid = wdata & 0x7;
            xtalid = (wdata >> 4) & 0x7;

            // check if the stripid and xtalid are in the allowed range and if not skip the rest of the block
            if (stripid < ElectronicsIdGPU::MIN_STRIPID || stripid > ElectronicsIdGPU::MAX_STRIPID ||
                xtalid < ElectronicsIdGPU::MIN_XTALID || xtalid > ElectronicsIdGPU::MAX_XTALID) {
              bad_block = true;
            }
            if (channel > 0) {
              // check if the stripid has increased or that the xtalid has increased from the previous data word. If not something is wrong and the rest of the block is skipped.
              auto const prev_channel = channel - 1;
              auto const prevwdata = current_tower_block[1 + prev_channel * 3];
              uint8_t const laststripid = prevwdata & 0x7;
              uint8_t const lastxtalid = (prevwdata >> 4) & 0x7;
              if ((stripid == laststripid && xtalid <= lastxtalid) || (stripid < laststripid)) {
                bad_block = true;
              }
            }
          }

          // check if this thread has the lowest bad block
          if (bad_block && channel < ch_with_bad_block) {
            alpaka::atomicMin(acc, &ch_with_bad_block, channel, alpaka::hierarchy::Threads{});
          }

          // make sure that all threads that have to have set the ch_with_bad_block shared memory
          alpaka::syncBlockThreads(acc);

          // threads outside of the range or bad block detected in this thread or one working on a lower block -> stop this loop iteration here
          if (channel >= nchannels || channel >= ch_with_bad_block) {
            continue;
          }

          ElectronicsIdGPU eid{fed2dcc(fed), ttid, stripid, xtalid};
          auto const didraw = isBarrel ? compute_ebdetid(eid) : eid2did[eid.linearIndex()].rawid();
          // skip channels with an invalid detid
          if (didraw == 0)
            continue;

          // get samples
          uint16_t sampleValues[kSampleSize];
          sampleValues[0] = (wdata >> 16) & 0x3fff;
          sampleValues[1] = (wdata >> 32) & 0x3fff;
          sampleValues[2] = (wdata >> 48) & 0x3fff;
          auto const wdata1 = current_tower_block[2 + channel * 3];
          sampleValues[3] = wdata1 & 0x3fff;
          sampleValues[4] = (wdata1 >> 16) & 0x3fff;
          sampleValues[5] = (wdata1 >> 32) & 0x3fff;
          sampleValues[6] = (wdata1 >> 48) & 0x3fff;
          auto const wdata2 = current_tower_block[3 + channel * 3];
          sampleValues[7] = wdata2 & 0x3fff;
          sampleValues[8] = (wdata2 >> 16) & 0x3fff;
          sampleValues[9] = (wdata2 >> 32) & 0x3fff;

          // check gain
          bool isSaturation = true;
          short firstGainZeroSampID{-1}, firstGainZeroSampADC{-1};
          for (uint32_t si = 0; si < kSampleSize; ++si) {
            if (gainId(sampleValues[si]) == 0) {
              firstGainZeroSampID = si;
              firstGainZeroSampADC = adc(sampleValues[si]);
              break;
            }
          }
          if (firstGainZeroSampID != -1) {
            unsigned int plateauEnd = std::min(kSampleSize, (unsigned int)(firstGainZeroSampID + 5));
            for (unsigned int s = firstGainZeroSampID; s < plateauEnd; s++) {
              if (!(gainId(sampleValues[s]) == 0 && adc(sampleValues[s]) == firstGainZeroSampADC)) {
                isSaturation = false;
                break;
              }  //it's not saturation
            }
            // get rid of channels which are stuck in gain0
            if (firstGainZeroSampID < 3) {
              isSaturation = false;
            }
            if (!isSaturation)
              continue;
          } else {  // there is no zero gainId sample
            // gain switch check
            short numGain = 1;
            bool gainSwitchError = false;
            for (unsigned int si = 1; si < kSampleSize; ++si) {
              if ((gainId(sampleValues[si - 1]) > gainId(sampleValues[si])) && numGain < 5)
                gainSwitchError = true;
              if (gainId(sampleValues[si - 1]) == gainId(sampleValues[si]))
                numGain++;
              else
                numGain = 1;
            }
            if (gainSwitchError)
              continue;
          }

          auto const pos = alpaka::atomicAdd(acc, pChannelsCounter, 1u, alpaka::hierarchy::Threads{});

          // store to global
          ids[pos] = didraw;
          std::memcpy(&samples[pos * kSampleSize], sampleValues, kSampleSize * sizeof(uint16_t));
        }

        current_tower_block += block_length;
      }
    }

  private:
    ALPAKA_FN_INLINE ALPAKA_FN_ACC void print_raw_buffer(uint8_t const* const buffer,
                                                         uint32_t const nbytes,
                                                         uint32_t const nbytes_per_row = 20) const {
      for (uint32_t i = 0; i < nbytes; ++i) {
        if (i % nbytes_per_row == 0 && i > 0)
          printf("\n");
        printf("%02X ", buffer[i]);
      }
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC void print_first3bits(uint64_t const* buffer, uint32_t size) const {
      for (uint32_t i = 0; i < size; ++i) {
        uint8_t const b61 = (buffer[i] >> 61) & 0x1;
        uint8_t const b62 = (buffer[i] >> 62) & 0x1;
        uint8_t const b63 = (buffer[i] >> 63) & 0x1;
        printf("[word: %u] %u%u%u\n", i, b63, b62, b61);
      }
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC bool is_barrel(uint8_t dccid) const {
      return dccid >= ElectronicsIdGPU::MIN_DCCID_EBM && dccid <= ElectronicsIdGPU::MAX_DCCID_EBP;
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC uint8_t fed2dcc(int fed) const { return static_cast<uint8_t>(fed - 600); }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC int zside_for_eb(ElectronicsIdGPU const& eid) const {
      int dcc = eid.dccId();
      return ((dcc >= ElectronicsIdGPU::MIN_DCCID_EBM && dcc <= ElectronicsIdGPU::MAX_DCCID_EBM)) ? -1 : 1;
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC uint8_t find_next_tower_block(uint64_t const*& current_tower_block,
                                                                 uint64_t const* trailer,
                                                                 uint32_t const bx,
                                                                 uint32_t const lv1) const {
      const auto* next_tower_block = current_tower_block + 1;  // move forward to skip the broken header

      // expected LV1, BX, #TS
      const uint64_t lv1local = ((lv1 - 1) & TOWER_L1_MASK);
      const uint64_t bxlocal = (bx != 3564) ? bx : 0;
      // The CPU unpacker also checks the # time samples expected in the header
      // but those are currently not available here

      // construct tower header and mask
      const uint64_t sign = 0xC0000000C0000000 + (lv1local << TOWER_L1_B) + (bxlocal << TOWER_BX_B);
      const uint64_t mask =
          0xC0001000D0000000 + (uint64_t(TOWER_L1_MASK) << TOWER_L1_B) + (uint64_t(TOWER_BX_MASK) << TOWER_BX_B);

      while (next_tower_block < trailer) {
        if ((*next_tower_block & mask) == sign) {
          current_tower_block = next_tower_block;
          return uint8_t(*next_tower_block & TOWER_ID_MASK);
        } else {
          ++next_tower_block;
        }
      }
      return TOWER_ID_MASK;  // return the maximum value
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC bool is_synced_towerblock(uint16_t const dccbx,
                                                             uint16_t const bx,
                                                             uint16_t const dccl1,
                                                             uint16_t const l1) const {
      bool const bxsync = (bx == 0 && dccbx == 3564) || (bx == dccbx && dccbx != 3564);
      bool const l1sync = (l1 == ((dccl1 - 1) & 0xfff));
      return bxsync && l1sync;
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC bool right_tower_for_eb(int tower) const {
      // for EB, two types of tower (LVRB top/bottom)
      return (tower > 12 && tower < 21) || (tower > 28 && tower < 37) || (tower > 44 && tower < 53) ||
             (tower > 60 && tower < 69);
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC uint32_t compute_ebdetid(ElectronicsIdGPU const& eid) const {
      // as in Geometry/EcalMaping/.../EcalElectronicsMapping
      auto const dcc = eid.dccId();
      auto const tower = eid.towerId();
      auto const strip = eid.stripId();
      auto const xtal = eid.xtalId();

      int smid = 0;
      int iphi = 0;
      bool EBPlus = (zside_for_eb(eid) > 0);
      bool EBMinus = !EBPlus;

      if (zside_for_eb(eid) < 0) {
        smid = dcc + 19 - ElectronicsIdGPU::DCCID_PHI0_EBM;
        iphi = (smid - 19) * ElectronicsIdGPU::kCrystalsInPhi;
        iphi += 5 * ((tower - 1) % ElectronicsIdGPU::kTowersInPhi);
      } else {
        smid = dcc + 1 - ElectronicsIdGPU::DCCID_PHI0_EBP;
        iphi = (smid - 1) * ElectronicsIdGPU::kCrystalsInPhi;
        iphi += 5 * (ElectronicsIdGPU::kTowersInPhi - ((tower - 1) % ElectronicsIdGPU::kTowersInPhi) - 1);
      }

      bool RightTower = right_tower_for_eb(tower);
      int ieta = 5 * ((tower - 1) / ElectronicsIdGPU::kTowersInPhi) + 1;
      if (RightTower) {
        ieta += (strip - 1);
        if (strip % 2 == 1) {
          if (EBMinus)
            iphi += (xtal - 1) + 1;
          else
            iphi += (4 - (xtal - 1)) + 1;
        } else {
          if (EBMinus)
            iphi += (4 - (xtal - 1)) + 1;
          else
            iphi += (xtal - 1) + 1;
        }
      } else {
        ieta += 4 - (strip - 1);
        if (strip % 2 == 1) {
          if (EBMinus)
            iphi += (4 - (xtal - 1)) + 1;
          else
            iphi += (xtal - 1) + 1;
        } else {
          if (EBMinus)
            iphi += (xtal - 1) + 1;
          else
            iphi += (4 - (xtal - 1)) + 1;
        }
      }

      if (zside_for_eb(eid) < 0)
        ieta = -ieta;

      DetId did{DetId::Ecal, EcalBarrel};
      return did.rawId() | ((ieta > 0) ? (0x10000 | (ieta << 9)) : ((-ieta) << 9)) | (iphi & 0x1FF);
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC int adc(uint16_t sample) const { return sample & 0xfff; }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC int gainId(uint16_t sample) const { return (sample >> 12) & 0x3; }
  };

  void unpackRaw(Queue& queue,
                 InputDataHost const& inputHost,
                 EcalDigiDeviceCollection& digisDevEB,
                 EcalDigiDeviceCollection& digisDevEE,
                 EcalElectronicsMappingDevice const& mapping,
                 uint32_t const nfedsWithData,
                 uint32_t const nbytesTotal) {
    // input device buffers
    ecal::raw::InputDataDevice inputDevice(queue, nbytesTotal, nfedsWithData);

    // transfer the raw data
    alpaka::memcpy(queue, inputDevice.data, inputHost.data);
    alpaka::memcpy(queue, inputDevice.offsets, inputHost.offsets);
    alpaka::memcpy(queue, inputDevice.feds, inputHost.feds);

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(nfedsWithData, 32);  // 32 channels per block
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        Kernel_unpack{},
                        inputDevice.data.data(),
                        inputDevice.offsets.data(),
                        inputDevice.feds.data(),
                        digisDevEB.view(),
                        digisDevEE.view(),
                        mapping.const_view(),
                        nbytesTotal);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::raw
