#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "EventFilter/HcalRawToDigi/plugins/DecodeGPU.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

namespace hcal {
  namespace raw {

    __forceinline__ __device__ char const* get_subdet_str(DetId const& did) {
      switch (did.subdetId()) {
        case HcalEmpty:
          return "HcalEmpty";
          break;
        case HcalBarrel:
          return "HcalBarrel";
          break;
        case HcalEndcap:
          return "HcalEndcap";
          break;
        case HcalOuter:
          return "HcalOuter";
          break;
        case HcalForward:
          return "HcalForward";
          break;
        case HcalTriggerTower:
          return "HcalTriggerTower";
          break;
        case HcalOther:
          return "HcalOther";
          break;
        default:
          return "Unknown";
          break;
      }

      return "Unknown";
    }

    __forceinline__ __device__ bool is_channel_header_word(uint16_t const* ptr) {
      uint8_t bit = (*ptr >> 15) & 0x1;
      return bit == 1;
    }

    template <typename T>
    constexpr bool is_power_of_two(T x) {
      return (x != 0) && ((x & (x - 1)) == 0);
    }

    template <int NTHREADS>
    __global__ void kernel_rawdecode_test(unsigned char const* data,
                                          uint32_t const* offsets,
                                          int const* feds,
                                          uint32_t const* eid2did,
                                          uint32_t const* eid2tid,
                                          uint16_t* digisF01HE,
                                          uint32_t* idsF01HE,
                                          uint16_t* digisF5HB,
                                          uint32_t* idsF5HB,
                                          uint8_t* npresamplesF5HB,
                                          uint16_t* digisF3HB,
                                          uint32_t* idsF3HB,
                                          uint32_t* pChannelsCounters,
                                          uint32_t const nsamplesF01HE,
                                          uint32_t const nsamplesF5HB,
                                          uint32_t const nsamplesF3HB,
                                          uint32_t const nBytesTotal) {
      // in order to properly use cooperative groups
      static_assert(is_power_of_two(NTHREADS) == true && NTHREADS <= 32);

      thread_block_tile<NTHREADS> thread_group = tiled_partition<NTHREADS>(this_thread_block());

      auto const iamc = threadIdx.x / NTHREADS;
      auto const ifed = blockIdx.x;
      auto const offset = offsets[ifed];

#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
      if (ifed > 0 || iamc > 0)
        return;
      printf("threadIdx.x = %d rank = %d iamc = %d\n", threadIdx.x, thread_group.thread_rank(), iamc);
#endif

#ifdef HCAL_RAWDECODE_GPUDEBUG
      auto const fed = feds[ifed];
      auto const size = ifed == gridDim.x - 1 ? nBytesTotal - offset : offsets[ifed + 1] - offset;
      printf("ifed = %d fed = %d offset = %u size = %u\n", ifed, fed, offset, size);
#endif

      // offset to the right raw buffer
      uint64_t const* buffer = reinterpret_cast<uint64_t const*>(data + offset);

#ifdef HCAL_RAWDECODE_GPUDEBUG
      //
      // fed header
      //
      auto const fed_header = buffer[0];
      uint32_t const fed_id = (fed_header >> 8) & 0xfff;
      uint32_t const bx = (fed_header >> 20) & 0xfff;
      uint32_t const lv1 = (fed_header >> 32) & 0xffffff;
      uint8_t const trigger_type = (fed_header >> 56) & 0xf;
      uint8_t const bid_fed_header = (fed_header >> 60) & 0xf;

      printf("fed = %d fed_id = %u bx = %u lv1 = %u trigger_type = %u bid = %u\n",
             fed,
             fed_id,
             bx,
             lv1,
             trigger_type,
             bid_fed_header);
#endif

      // amc 13 header
      auto const amc13word = buffer[1];
      uint8_t const namc = (amc13word >> 52) & 0xf;
      if (iamc >= namc)
        return;

#ifdef HCAL_RAWDECODE_GPUDEBUG
      uint8_t const amc13version = (amc13word >> 60) & 0xf;
      uint32_t const amc13OrbitNumber = (amc13word >> 4) & 0xffffffffu;
      printf("fed = %d namc = %u amc13version = %u amc13OrbitNumber = %u\n", fed, namc, amc13version, amc13OrbitNumber);
#endif

      // compute the offset int to the right buffer
      uint32_t amcoffset = 0;
      for (uint8_t ii = 0u; ii < iamc; ii++) {
        auto const word = buffer[2 + ii];
        int const amcSize = (word >> 32) & 0xffffff;
        amcoffset += amcSize;
      }

      auto const word = buffer[2 + iamc];
      int const amcSize = (word >> 32) & 0xffffff;

#ifdef HCAL_RAWDECODE_GPUDEBUG
      uint16_t const amcid = word & 0xffff;
      int const slot = (word >> 16) & 0xf;
      int const amcBlockNumber = (word >> 20) & 0xff;
      printf("fed = %d amcid = %u slot = %d amcBlockNumber = %d\n", fed, amcid, slot, amcBlockNumber);

      bool const amcmore = ((word >> 61) & 0x1) != 0;
      bool const amcSegmented = ((word >> 60) & 0x1) != 0;
      bool const amcLengthOk = ((word >> 62) & 0x1) != 0;
      bool const amcCROk = ((word >> 56) & 0x1) != 0;
      bool const amcDataPresent = ((word >> 58) & 0x1) != 0;
      bool const amcDataValid = ((word >> 56) & 0x1) != 0;
      bool const amcEnabled = ((word >> 59) & 0x1) != 0;
      printf(
          "fed = %d amcmore = %d amcSegmented = %d, amcLengthOk = %d amcCROk = %d\n>> amcDataPresent = %d amcDataValid "
          "= %d amcEnabled = %d\n",
          fed,
          static_cast<int>(amcmore),
          static_cast<int>(amcSegmented),
          static_cast<int>(amcLengthOk),
          static_cast<int>(amcCROk),
          static_cast<int>(amcDataPresent),
          static_cast<int>(amcDataValid),
          static_cast<int>(amcEnabled));
#endif

      // get to the payload
      auto const* payload64 = buffer + 2 + namc + amcoffset;

#ifdef HCAL_RAWDECODE_GPUDEBUG
      // uhtr header v1 1st 64 bits
      auto const payload64_w0 = payload64[0];
#endif
      // uhtr n bytes comes from amcSize, according to the cpu version!
      uint32_t const data_length64 = amcSize;

#ifdef HCAL_RAWDECODE_GPUDEBUG
      uint16_t bcn = (payload64_w0 >> 20) & 0xfff;
      uint32_t evn = (payload64_w0 >> 32) & 0xffffff;
      printf("fed = %d data_length64 = %u bcn = %u evn = %u\n", fed, data_length64, bcn, evn);
#endif

      // uhtr header v1 2nd 64 bits
      auto const payload64_w1 = payload64[1];
      uint8_t const uhtrcrate = payload64_w1 & 0xff;
      uint8_t const uhtrslot = (payload64_w1 >> 8) & 0xf;
      uint8_t const presamples = (payload64_w1 >> 12) & 0xf;
      uint8_t const payloadFormat = (payload64_w1 >> 44) & 0xf;

#ifdef HCAL_RAWDECODE_GPUDEBUG
      uint16_t const orbitN = (payload64_w1 >> 16) & 0xffff;
      uint8_t const firmFlavor = (payload64_w1 >> 32) & 0xff;
      uint8_t const eventType = (payload64_w1 >> 40) & 0xf;
      printf(
          "fed = %d crate = %u slot = %u presamples = %u\n>>> orbitN = %u firmFlavor = %u eventType = %u payloadFormat "
          "= %u\n",
          fed,
          uhtrcrate,
          uhtrslot,
          presamples,
          orbitN,
          firmFlavor,
          eventType,
          payloadFormat);
#endif

      // this should be filtering out uMNio...
      if (payloadFormat != 1)
        return;

      // skip uhtr header words
      auto const channelDataSize = data_length64 - 2;        // 2 uhtr header v1 words
      auto const* channelDataBuffer64Start = payload64 + 2;  // 2 uhtr header v2 wds
      auto const* ptr = reinterpret_cast<uint16_t const*>(channelDataBuffer64Start);
      auto const* end = ptr + sizeof(uint64_t) / sizeof(uint16_t) * (channelDataSize - 1);
      auto const t_rank = thread_group.thread_rank();

      // iterate through the channel data
      while (ptr != end) {
        // this is the starting point for this thread group for this iteration
        // with respect to this pointer every thread will move forward afterwards
        auto const* const start_ptr = ptr;

#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
        thread_group.sync();
#endif

        // skip to the header word of the right channel for this thread
        int counter = 0;
        while (counter < thread_group.thread_rank()) {
          // just a check for threads that land beyond the end
          if (ptr == end)
            break;

          // move ptr one forward past header
          if (is_channel_header_word(ptr))
            ++ptr;
          else {
            // go to the next channel and do not consider this guy as a channel
            while (ptr != end)
              if (!is_channel_header_word(ptr))
                ++ptr;
              else
                break;
            continue;
          }

          // skip
          while (ptr != end)
            if (!is_channel_header_word(ptr))
              ++ptr;
            else
              break;
          counter++;
        }

#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
        thread_group.sync();
        printf("ptr - start_ptr = %d counter = %d rank = %d\n", static_cast<int>(ptr - start_ptr), counter, t_rank);
#endif

        // when the end is near, channels will land outside of the [start_ptr, end) region
        if (ptr != end) {
          // for all of the flavors, these 2 guys have the same bit layout
          uint8_t const flavor = (ptr[0] >> 12) & 0x7;
          uint8_t const channelid = ptr[0] & 0xff;
          auto const* const new_channel_start = ptr;

          // flavor dependent stuff
          switch (flavor) {
            case 0:
            case 1: {
              // treat eid and did
              uint8_t fiber = (channelid >> 3) & 0x1f;
              uint8_t fchannel = channelid & 0x7;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, fiber, fchannel, false};
              auto const did = HcalDetId{eid2did[eid.linearIndex()]};

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u subdet = %s\n",
                     eid.rawId(),
                     eid.linearIndex(),
                     did.rawId(),
                     get_subdet_str(did));
              printf("flavor = %u crate = %u slot = %u channelid = %u fiber = %u fchannel = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     fiber,
                     fchannel);
#endif

              // remove digis not for HE
              if (did.subdetId() != HcalEndcap)
                break;

              // count words
              auto const* channel_header_word = ptr++;
              while (!is_channel_header_word(ptr) && ptr != end)
                ++ptr;
              auto const* channel_end = ptr;  // set ptr
              uint32_t const nwords = channel_end - channel_header_word;

              // filter out this digi if nwords does not equal expected
              auto const expected_words = compute_stride<Flavor1>(nsamplesF01HE);
              if (nwords != expected_words)
                break;

              // inc the number of digis of this type
              auto const pos = atomicAdd(&pChannelsCounters[OutputF01HE], 1);
#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
              printf("rank = %d pos = %d\n", thread_group.thread_rank(), pos);
#endif

              // store to global mem words for this digi
              idsF01HE[pos] = did.rawId();

              for (uint32_t iword = 0; iword < expected_words; iword++)
                digisF01HE[pos * expected_words + iword] = channel_header_word[iword];

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("nwords = %u\n", nwords);
#endif

              break;
            }
            case 3: {
              // treat eid and did
              uint8_t fiber = (channelid >> 3) & 0x1f;
              uint8_t fchannel = channelid & 0x7;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, fiber, fchannel, false};
              auto const did = HcalDetId{eid2did[eid.linearIndex()]};

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u subdet = %s\n",
                     eid.rawId(),
                     eid.linearIndex(),
                     did.rawId(),
                     get_subdet_str(did));
              printf("flavor = %u crate = %u slot = %u channelid = %u fiber = %u fchannel = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     fiber,
                     fchannel);
#endif

              // remove digis not for HE
              if (did.subdetId() != HcalBarrel)
                break;

              // count words
              auto const* channel_header_word = ptr++;
              while (!is_channel_header_word(ptr) && ptr != end)
                ++ptr;
              auto const* channel_end = ptr;  // set ptr
              uint32_t const nwords = channel_end - channel_header_word;

              // filter out this digi if nwords does not equal expected
              auto const expected_words = compute_stride<Flavor3>(nsamplesF3HB);
              if (nwords != expected_words)
                break;

              // inc the number of digis of this type
              auto const pos = atomicAdd(&pChannelsCounters[OutputF3HB], 1);

              // store to global mem words for this digi
              idsF3HB[pos] = did.rawId();
              for (uint32_t iword = 0; iword < expected_words; iword++)
                digisF3HB[pos * expected_words + iword] = channel_header_word[iword];

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("nwords = %u\n", nwords);
#endif

              break;
            }
            case 2: {
              uint8_t fiber = (channelid >> 3) & 0x1f;
              uint8_t fchannel = channelid & 0x7;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, fiber, fchannel, false};
              auto const did = DetId{eid2did[eid.linearIndex()]};

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u subdet = %s\n",
                     eid.rawId(),
                     eid.linearIndex(),
                     did.rawId(),
                     get_subdet_str(did));
              printf("flavor = %u crate = %u slot = %u channelid = %u fiber = %u fchannel = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     fiber,
                     fchannel);
#endif

              break;
            }
            case 4: {
              uint8_t link = (channelid >> 4) & 0xf;
              uint8_t tower = channelid & 0xf;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, link, tower, true};
              auto const did = DetId{eid2tid[eid.linearIndex()]};

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u subdet = %s\n",
                     eid.rawId(),
                     eid.linearIndex(),
                     did.rawId(),
                     get_subdet_str(did));
              printf("flavor = %u crate = %u slot = %u channelid = %u link = %u tower = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     link,
                     tower);
#endif

              break;
            }
            case 5: {
              uint8_t fiber = (channelid >> 2) & 0x3f;
              uint8_t fchannel = channelid & 0x3;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, fiber, fchannel, false};
              auto const did = HcalDetId{eid2did[eid.linearIndex()]};

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u subdet = %s\n",
                     eid.rawId(),
                     eid.linearIndex(),
                     did.rawId(),
                     get_subdet_str(did));
              printf("flavor = %u crate = %u slot = %u channelid = %u fiber = %u fchannel = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     fiber,
                     fchannel);
#endif

              // remove digis not for HB
              if (did.subdetId() != HcalBarrel)
                break;

              // count words
              auto const* channel_header_word = ptr++;
              while (!is_channel_header_word(ptr) && ptr != end)
                ++ptr;
              auto const* channel_end = ptr;  // set ptr
              uint32_t const nwords = channel_end - channel_header_word;

              // filter out this digi if nwords does not equal expected
              auto const expected_words = compute_stride<Flavor5>(nsamplesF5HB);
              if (nwords != expected_words)
                break;

              // inc the number of digis of this type
              auto const pos = atomicAdd(&pChannelsCounters[OutputF5HB], 1);

#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
              printf("rank = %d pos = %d\n", thread_group.thread_rank(), pos);
#endif

              // store to global mem words for this digi
              idsF5HB[pos] = did.rawId();
              npresamplesF5HB[pos] = presamples;
              for (uint32_t iword = 0; iword < expected_words; iword++)
                digisF5HB[pos * expected_words + iword] = channel_header_word[iword];

              break;
            }
            case 7: {
              uint8_t const fiber = (channelid >> 2) & 0x3f;
              uint8_t const fchannel = channelid & 0x3;
              HcalElectronicsId eid{uhtrcrate, uhtrslot, fiber, fchannel, false};
              auto const did = DetId{eid2did[eid.linearIndex()]};

              /* uncomment to check the linear index validity
              if (eid.rawId() >= HcalElectronicsId::maxLinearIndex) {
#ifdef HCAL_RAWDECODE_GPUDEBUG
                  printf("*** rawid = %u has no known det id***\n", eid.rawId());
#endif
                  break;
              }
              */

#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("erawId = %u linearIndex = %u drawid = %u\n", eid.rawId(), eid.linearIndex(), did.rawId());
              printf("flavor = %u crate = %u slot = %u channelid = %u fiber = %u fchannel = %u\n",
                     flavor,
                     uhtrcrate,
                     uhtrslot,
                     channelid,
                     fiber,
                     fchannel);
#endif

              break;
            }
            default:
#ifdef HCAL_RAWDECODE_GPUDEBUG
              printf("flavor = %u crate = %u slot = %u channelid = %u\n", flavor, uhtrcrate, uhtrslot, channelid);
#endif
              ;
          }

          // skip to the next word in case
          // 1) current channel was not treated
          // 2) we are in the middle of the digi words and not at the end
          while (new_channel_start == ptr || !is_channel_header_word(ptr) && ptr != end)
            ++ptr;
        }

        // thread with rank 31 of the group will have the ptr pointing to the
        // header word of the next channel or the end
        int const offset_to_shuffle = ptr - start_ptr;

        // always receive from the last guy in the group
        auto const offset_for_rank31 = thread_group.shfl(offset_to_shuffle, NTHREADS - 1);

#ifdef HCAL_RAWDECODE_GPUDEBUG_CG
        printf("rank = %d offset_to_shuffle = %d offset_for_rank32 = %d\n",
               thread_group.thread_rank(),
               offset_to_shuffle,
               offset_for_rank31);
#endif

        // update the ptr for all threads of this group
        // NOTE: relative to the start_ptr that is the same for all threads of
        // this group
        ptr = start_ptr + offset_for_rank31;
      }
    }

    void entryPoint(InputDataCPU const& inputCPU,
                    InputDataGPU& inputGPU,
                    OutputDataGPU& outputGPU,
                    ScratchDataGPU& scratchGPU,
                    OutputDataCPU& outputCPU,
                    ConditionsProducts const& conditions,
                    ConfigurationParameters const& config,
                    cudaStream_t cudaStream,
                    uint32_t const nfedsWithData,
                    uint32_t const nbytesTotal) {
      // transfer
      cudaCheck(cudaMemcpyAsync(inputGPU.data.get(),
                                inputCPU.data.get(),
                                nbytesTotal * sizeof(unsigned char),
                                cudaMemcpyHostToDevice,
                                cudaStream));
      cudaCheck(cudaMemcpyAsync(inputGPU.offsets.get(),
                                inputCPU.offsets.get(),
                                nfedsWithData * sizeof(uint32_t),
                                cudaMemcpyHostToDevice,
                                cudaStream));
      cudaCheck(
          cudaMemsetAsync(scratchGPU.pChannelsCounters.get(), 0, sizeof(uint32_t) * numOutputCollections, cudaStream));
      cudaCheck(cudaMemcpyAsync(
          inputGPU.feds.get(), inputCPU.feds.get(), nfedsWithData * sizeof(int), cudaMemcpyHostToDevice, cudaStream));

      // 12 is the max number of modules per crate
      kernel_rawdecode_test<32><<<nfedsWithData, 12 * 32, 0, cudaStream>>>(inputGPU.data.get(),
                                                                           inputGPU.offsets.get(),
                                                                           inputGPU.feds.get(),
                                                                           conditions.eMappingProduct.eid2did,
                                                                           conditions.eMappingProduct.eid2tid,
                                                                           outputGPU.digisF01HE.data.get(),
                                                                           outputGPU.digisF01HE.ids.get(),
                                                                           outputGPU.digisF5HB.data.get(),
                                                                           outputGPU.digisF5HB.ids.get(),
                                                                           outputGPU.digisF5HB.npresamples.get(),
                                                                           outputGPU.digisF3HB.data.get(),
                                                                           outputGPU.digisF3HB.ids.get(),
                                                                           scratchGPU.pChannelsCounters.get(),
                                                                           config.nsamplesF01HE,
                                                                           config.nsamplesF5HB,
                                                                           config.nsamplesF3HB,
                                                                           nbytesTotal);
      cudaCheck(cudaGetLastError());

      cudaCheck(cudaMemcpyAsync(outputCPU.nchannels.get(),
                                scratchGPU.pChannelsCounters.get(),
                                sizeof(uint32_t) * numOutputCollections,
                                cudaMemcpyDeviceToHost,
                                cudaStream));
    }

  }  // namespace raw
}  // namespace hcal
