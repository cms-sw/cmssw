#include "EventFilter/EcalRawToDigi/interface/ElectronicsIdGPU.h"

#include "UnpackGPU.h"

namespace ecal {
  namespace raw {

    __forceinline__ __device__ void print_raw_buffer(uint8_t const* const buffer,
                                                     uint32_t const nbytes,
                                                     uint32_t const nbytes_per_row = 20) {
      for (uint32_t i = 0; i < nbytes; i++) {
        if (i % nbytes_per_row == 0 && i > 0)
          printf("\n");
        printf("%02X ", buffer[i]);
      }
    }

    __forceinline__ __device__ void print_first3bits(uint64_t const* buffer, uint32_t size) {
      for (uint32_t i = 0; i < size; ++i) {
        uint8_t const b61 = (buffer[i] >> 61) & 0x1;
        uint8_t const b62 = (buffer[i] >> 62) & 0x1;
        uint8_t const b63 = (buffer[i] >> 63) & 0x1;
        printf("[word: %u] %u%u%u\n", i, b63, b62, b61);
      }
    }

    __forceinline__ __device__ bool is_barrel(uint8_t dccid) {
      return dccid >= ElectronicsIdGPU::MIN_DCCID_EBM && dccid <= ElectronicsIdGPU::MAX_DCCID_EBP;
    }

    __forceinline__ __device__ uint8_t fed2dcc(int fed) { return static_cast<uint8_t>(fed - 600); }

    __forceinline__ __device__ int zside_for_eb(ElectronicsIdGPU const& eid) {
      int dcc = eid.dccId();
      return ((dcc >= ElectronicsIdGPU::MIN_DCCID_EBM && dcc <= ElectronicsIdGPU::MAX_DCCID_EBM)) ? -1 : 1;
    }

    __forceinline__ __device__ bool is_synced_towerblock(uint16_t const dccbx,
                                                         uint16_t const bx,
                                                         uint16_t const dccl1,
                                                         uint16_t const l1) {
      bool const bxsync = (bx == 0 && dccbx == 3564) || (bx == dccbx && dccbx != 3564);
      bool const l1sync = (l1 == ((dccl1 - 1) & 0xfff));
      return bxsync && l1sync;
    }

    __forceinline__ __device__ bool right_tower_for_eb(int tower) {
      // for EB, two types of tower (LVRB top/bottom)
      if ((tower > 12 && tower < 21) || (tower > 28 && tower < 37) || (tower > 44 && tower < 53) ||
          (tower > 60 && tower < 69))
        return true;
      else
        return false;
    }

    __forceinline__ __device__ uint32_t compute_ebdetid(ElectronicsIdGPU const& eid) {
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

    __forceinline__ __device__ int adc(uint16_t sample) { return sample & 0xfff; }

    __forceinline__ __device__ int gainId(uint16_t sample) { return (sample >> 12) & 0x3; }

    template <int NTHREADS>
    __global__ void kernel_unpack_test(unsigned char const* __restrict__ data,
                                       uint32_t const* __restrict__ offsets,
                                       int const* __restrict__ feds,
                                       uint16_t* samplesEB,
                                       uint16_t* samplesEE,
                                       uint32_t* idsEB,
                                       uint32_t* idsEE,
                                       uint32_t* pChannelsCounterEBEE,
                                       uint32_t const* eid2did,
                                       uint32_t const nbytesTotal) {
      // indices
      auto const ifed = blockIdx.x;

      // offset in bytes
      auto const offset = offsets[ifed];
      // fed id
      auto const fed = feds[ifed];
      auto const isBarrel = is_barrel(static_cast<uint8_t>(fed - 600));
      // size
      auto const size = ifed == gridDim.x - 1 ? nbytesTotal - offset : offsets[ifed + 1] - offset;
      auto* samples = isBarrel ? samplesEB : samplesEE;
      auto* ids = isBarrel ? idsEB : idsEE;
      auto* pChannelsCounter = isBarrel ? &pChannelsCounterEBEE[0] : &pChannelsCounterEBEE[1];

      // offset to the right raw buffer
      uint64_t const* buffer = reinterpret_cast<uint64_t const*>(data + offset);

      // dump first 3 bits for each 64-bit word
      //print_first3bits(buffer, size / 8);

      //
      // fed header
      //
      auto const fed_header = buffer[0];
      uint32_t bx = (fed_header >> 20) & 0xfff;
      uint32_t lv1 = (fed_header >> 32) & 0xffffff;

      // 9 for fed + dcc header
      // 36 for 4 EE TCC blocks or 18 for 1 EB TCC block
      // 6 for SR block size

      // dcc header w2
      auto const w2 = buffer[2];
      uint8_t const fov = (w2 >> 48) & 0xf;

      //
      // print Tower block headers
      //
      uint8_t ntccblockwords = isBarrel ? 18 : 36;
      auto const* tower_blocks_start = buffer + 9 + ntccblockwords + 6;
      auto const* trailer = buffer + (size / 8 - 1);
      auto const* current_tower_block = tower_blocks_start;
      while (current_tower_block != trailer) {
        auto const w = *current_tower_block;
        uint8_t ttid = w & 0xff;
        uint16_t bxlocal = (w >> 16) & 0xfff;
        uint16_t lv1local = (w >> 32) & 0xfff;
        uint16_t block_length = (w >> 48) & 0x1ff;

        uint16_t const dccbx = bx & 0xfff;
        uint16_t const dccl1 = lv1 & 0xfff;
        // fov>=1 is required to support simulated data for which bx==bxlocal==0
        if (fov >= 1 && !is_synced_towerblock(dccbx, bxlocal, dccl1, lv1local)) {
          current_tower_block += block_length;
          continue;
        }

        // go through all the channels
        // get the next channel coordinates
        uint32_t nchannels = (block_length - 1) / 3;

        // 1 threads per channel in this block
        for (uint32_t ich = 0; ich < nchannels; ich += NTHREADS) {
          auto const i_to_access = ich + threadIdx.x;
          // threads outside of the range -> leave the loop
          if (i_to_access >= nchannels)
            break;

          // inc the channel's counter and get the pos where to store
          auto const wdata = current_tower_block[1 + i_to_access * 3];
          uint8_t const stripid = wdata & 0x7;
          uint8_t const xtalid = (wdata >> 4) & 0x7;
          ElectronicsIdGPU eid{fed2dcc(fed), ttid, stripid, xtalid};
          auto const didraw = isBarrel ? compute_ebdetid(eid) : eid2did[eid.linearIndex()];
          // FIXME: what kind of channels are these guys
          if (didraw == 0)
            continue;

          // get samples
          uint16_t sampleValues[10];
          sampleValues[0] = (wdata >> 16) & 0x3fff;
          sampleValues[1] = (wdata >> 32) & 0x3fff;
          sampleValues[2] = (wdata >> 48) & 0x3fff;
          auto const wdata1 = current_tower_block[2 + i_to_access * 3];
          sampleValues[3] = wdata1 & 0x3fff;
          sampleValues[4] = (wdata1 >> 16) & 0x3fff;
          sampleValues[5] = (wdata1 >> 32) & 0x3fff;
          sampleValues[6] = (wdata1 >> 48) & 0x3fff;
          auto const wdata2 = current_tower_block[3 + i_to_access * 3];
          sampleValues[7] = wdata2 & 0x3fff;
          sampleValues[8] = (wdata2 >> 16) & 0x3fff;
          sampleValues[9] = (wdata2 >> 32) & 0x3fff;

          // check gain
          bool isSaturation = true;
          short firstGainZeroSampID{-1}, firstGainZeroSampADC{-1};
          for (uint32_t si = 0; si < 10; si++) {
            if (gainId(sampleValues[si]) == 0) {
              firstGainZeroSampID = si;
              firstGainZeroSampADC = adc(sampleValues[si]);
              break;
            }
          }
          if (firstGainZeroSampID != -1) {
            unsigned int plateauEnd = std::min(10u, (unsigned int)(firstGainZeroSampID + 5));
            for (unsigned int s = firstGainZeroSampID; s < plateauEnd; s++) {
              if (gainId(sampleValues[s]) == 0 && adc(sampleValues[s]) == firstGainZeroSampADC) {
                ;
              } else {
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
            for (unsigned int si = 1; si < 10; si++) {
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

          auto const pos = atomicAdd(pChannelsCounter, 1);

          // store to global
          ids[pos] = didraw;
          samples[pos * 10] = sampleValues[0];
          samples[pos * 10 + 1] = sampleValues[1];
          samples[pos * 10 + 2] = sampleValues[2];
          samples[pos * 10 + 3] = sampleValues[3];
          samples[pos * 10 + 4] = sampleValues[4];
          samples[pos * 10 + 5] = sampleValues[5];
          samples[pos * 10 + 6] = sampleValues[6];
          samples[pos * 10 + 7] = sampleValues[7];
          samples[pos * 10 + 8] = sampleValues[8];
          samples[pos * 10 + 9] = sampleValues[9];
        }

        current_tower_block += block_length;
      }
    }

    void entryPoint(InputDataCPU const& inputCPU,
                    InputDataGPU& inputGPU,
                    OutputDataGPU& outputGPU,
                    ScratchDataGPU& scratchGPU,
                    OutputDataCPU& outputCPU,
                    ConditionsProducts const& conditions,
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
      cudaCheck(cudaMemsetAsync(scratchGPU.pChannelsCounter.get(),
                                0,
                                sizeof(uint32_t) * 2,  // EB + EE
                                cudaStream));
      cudaCheck(cudaMemcpyAsync(
          inputGPU.feds.get(), inputCPU.feds.get(), nfedsWithData * sizeof(int), cudaMemcpyHostToDevice, cudaStream));

      kernel_unpack_test<32><<<nfedsWithData, 32, 0, cudaStream>>>(inputGPU.data.get(),
                                                                   inputGPU.offsets.get(),
                                                                   inputGPU.feds.get(),
                                                                   outputGPU.digisEB.data.get(),
                                                                   outputGPU.digisEE.data.get(),
                                                                   outputGPU.digisEB.ids.get(),
                                                                   outputGPU.digisEE.ids.get(),
                                                                   scratchGPU.pChannelsCounter.get(),
                                                                   conditions.eMappingProduct.eid2did,
                                                                   nbytesTotal);
      cudaCheck(cudaGetLastError());

      // transfer the counters for how many eb and ee channels we got
      cudaCheck(cudaMemcpyAsync(outputCPU.nchannels.get(),
                                scratchGPU.pChannelsCounter.get(),
                                sizeof(uint32_t) * 2,
                                cudaMemcpyDeviceToHost,
                                cudaStream));
    }

  }  // namespace raw
}  // namespace ecal
