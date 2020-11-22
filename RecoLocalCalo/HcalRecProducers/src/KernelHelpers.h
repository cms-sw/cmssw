#ifndef RecoLocalCalo_HcalRecProducers_src_KernelHelpers_h
#define RecoLocalCalo_HcalRecProducers_src_KernelHelpers_h

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

#include "DeclsForKernels.h"

namespace hcal {
  namespace reconstruction {

    // this is from HcalTimeSlew.
    // HcalTimeSlew are values that come in from ESProducer that takes them
    // from a python config. see DeclsForKernels for more explanation
    __forceinline__ __device__ float compute_time_slew_delay(float const fC,
                                                             float const tzero,
                                                             float const slope,
                                                             float const tmax) {
      auto const rawDelay = tzero + slope * std::log(fC);
      return rawDelay < 0 ? 0 : (rawDelay > tmax ? tmax : rawDelay);
    }

    // HcalQIEShapes are hardcoded in HcalQIEData.cc basically
    // + some logic to generate 128 and 256 value arrays...
    __constant__ float const qie8shape[129] = {
        -1,   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   16,
        18,   20,   22,   24,   26,   28,   31,   34,   37,   40,   44,   48,   52,   57,   62,   57,   62,
        67,   72,   77,   82,   87,   92,   97,   102,  107,  112,  117,  122,  127,  132,  142,  152,  162,
        172,  182,  192,  202,  217,  232,  247,  262,  282,  302,  322,  347,  372,  347,  372,  397,  422,
        447,  472,  497,  522,  547,  572,  597,  622,  647,  672,  697,  722,  772,  822,  872,  922,  972,
        1022, 1072, 1147, 1222, 1297, 1372, 1472, 1572, 1672, 1797, 1922, 1797, 1922, 2047, 2172, 2297, 2422,
        2547, 2672, 2797, 2922, 3047, 3172, 3297, 3422, 3547, 3672, 3922, 4172, 4422, 4672, 4922, 5172, 5422,
        5797, 6172, 6547, 6922, 7422, 7922, 8422, 9047, 9672, 10297};

    __constant__ float const qie11shape[257] = {
        -0.5,    0.5,     1.5,     2.5,     3.5,     4.5,     5.5,     6.5,     7.5,     8.5,     9.5,     10.5,
        11.5,    12.5,    13.5,    14.5,    15.5,    17.5,    19.5,    21.5,    23.5,    25.5,    27.5,    29.5,
        31.5,    33.5,    35.5,    37.5,    39.5,    41.5,    43.5,    45.5,    47.5,    49.5,    51.5,    53.5,
        55.5,    59.5,    63.5,    67.5,    71.5,    75.5,    79.5,    83.5,    87.5,    91.5,    95.5,    99.5,
        103.5,   107.5,   111.5,   115.5,   119.5,   123.5,   127.5,   131.5,   135.5,   139.5,   147.5,   155.5,
        163.5,   171.5,   179.5,   187.5,   171.5,   179.5,   187.5,   195.5,   203.5,   211.5,   219.5,   227.5,
        235.5,   243.5,   251.5,   259.5,   267.5,   275.5,   283.5,   291.5,   299.5,   315.5,   331.5,   347.5,
        363.5,   379.5,   395.5,   411.5,   427.5,   443.5,   459.5,   475.5,   491.5,   507.5,   523.5,   539.5,
        555.5,   571.5,   587.5,   603.5,   619.5,   651.5,   683.5,   715.5,   747.5,   779.5,   811.5,   843.5,
        875.5,   907.5,   939.5,   971.5,   1003.5,  1035.5,  1067.5,  1099.5,  1131.5,  1163.5,  1195.5,  1227.5,
        1259.5,  1291.5,  1355.5,  1419.5,  1483.5,  1547.5,  1611.5,  1675.5,  1547.5,  1611.5,  1675.5,  1739.5,
        1803.5,  1867.5,  1931.5,  1995.5,  2059.5,  2123.5,  2187.5,  2251.5,  2315.5,  2379.5,  2443.5,  2507.5,
        2571.5,  2699.5,  2827.5,  2955.5,  3083.5,  3211.5,  3339.5,  3467.5,  3595.5,  3723.5,  3851.5,  3979.5,
        4107.5,  4235.5,  4363.5,  4491.5,  4619.5,  4747.5,  4875.5,  5003.5,  5131.5,  5387.5,  5643.5,  5899.5,
        6155.5,  6411.5,  6667.5,  6923.5,  7179.5,  7435.5,  7691.5,  7947.5,  8203.5,  8459.5,  8715.5,  8971.5,
        9227.5,  9483.5,  9739.5,  9995.5,  10251.5, 10507.5, 11019.5, 11531.5, 12043.5, 12555.5, 13067.5, 13579.5,
        12555.5, 13067.5, 13579.5, 14091.5, 14603.5, 15115.5, 15627.5, 16139.5, 16651.5, 17163.5, 17675.5, 18187.5,
        18699.5, 19211.5, 19723.5, 20235.5, 20747.5, 21771.5, 22795.5, 23819.5, 24843.5, 25867.5, 26891.5, 27915.5,
        28939.5, 29963.5, 30987.5, 32011.5, 33035.5, 34059.5, 35083.5, 36107.5, 37131.5, 38155.5, 39179.5, 40203.5,
        41227.5, 43275.5, 45323.5, 47371.5, 49419.5, 51467.5, 53515.5, 55563.5, 57611.5, 59659.5, 61707.5, 63755.5,
        65803.5, 67851.5, 69899.5, 71947.5, 73995.5, 76043.5, 78091.5, 80139.5, 82187.5, 84235.5, 88331.5, 92427.5,
        96523.5, 100620,  104716,  108812,  112908};

    // Conditions are transferred once per IOV
    // Access is performed based on the det id which is converted to a linear index
    // 2 funcs below are taken from HcalTopology (reimplemented here).
    // Inputs are constants that are also taken from HcalTopology
    // but passed to the kernel as arguments using the HclaTopology itself
    constexpr int32_t IPHI_MAX = 72;

    __forceinline__ __device__ uint32_t did2linearIndexHB(
        uint32_t const didraw, int const maxDepthHB, int const firstHBRing, int const lastHBRing, int const nEtaHB) {
      HcalDetId did{didraw};
      uint32_t const value = (did.depth() - 1) + maxDepthHB * (did.iphi() - 1);
      return did.ieta() > 0 ? value + maxDepthHB * hcal::reconstruction::IPHI_MAX * (did.ieta() - firstHBRing)
                            : value + maxDepthHB * hcal::reconstruction::IPHI_MAX * (did.ieta() + lastHBRing + nEtaHB);
    }

    __forceinline__ __device__ uint32_t did2linearIndexHE(uint32_t const didraw,
                                                          int const maxDepthHE,
                                                          int const maxPhiHE,
                                                          int const firstHERing,
                                                          int const lastHERing,
                                                          int const nEtaHE) {
      HcalDetId did{didraw};
      uint32_t const value = (did.depth() - 1) + maxDepthHE * (did.iphi() - 1);
      return did.ieta() > 0 ? value + maxDepthHE * maxPhiHE * (did.ieta() - firstHERing)
                            : value + maxDepthHE * maxPhiHE * (did.ieta() + lastHERing + nEtaHE);
    }

    __forceinline__ __device__ uint32_t get_qiecoder_index(uint32_t const capid, uint32_t const range) {
      return capid * 4 + range;
    }

    __forceinline__ __device__ float compute_reco_correction_factor(float const par1,
                                                                    float const par2,
                                                                    float const par3,
                                                                    float const x) {
      return par3 * x * x + par2 * x + par1;
    }

    // compute the charge using the adc, qie type and the appropriate qie shape array
    __forceinline__ __device__ float compute_coder_charge(
        int const qieType, uint8_t const adc, uint8_t const capid, float const* qieOffsets, float const* qieSlopes) {
      auto const range = qieType == 0 ? (adc >> 5) & 0x3 : (adc >> 6) & 0x3;
      auto const* qieShapeToUse = qieType == 0 ? qie8shape : qie11shape;
      auto const nbins = qieType == 0 ? 32 : 64;
      auto const center = adc % nbins == nbins - 1 ? 0.5 * (3 * qieShapeToUse[adc] - qieShapeToUse[adc - 1])
                                                   : 0.5 * (qieShapeToUse[adc] + qieShapeToUse[adc + 1]);
      auto const index = get_qiecoder_index(capid, range);
      return (center - qieOffsets[index]) / qieSlopes[index];
    }

    // this is from
    //  https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HcalRecProducers/src/HBHEPhase1Reconstructor.cc#L140

    __forceinline__ __device__ float compute_diff_charge_gain(int const qieType,
                                                              uint8_t adc,
                                                              uint8_t const capid,
                                                              float const* qieOffsets,
                                                              float const* qieSlopes,
                                                              bool const isqie11) {
      constexpr uint32_t mantissaMaskQIE8 = 0x1fu;
      constexpr uint32_t mantissaMaskQIE11 = 0x3f;
      auto const mantissaMask = isqie11 ? mantissaMaskQIE11 : mantissaMaskQIE8;
      auto const q = compute_coder_charge(qieType, adc, capid, qieOffsets, qieSlopes);
      auto const mantissa = adc & mantissaMask;

      if (mantissa == 0u || mantissa == mantissaMask - 1u)
        return compute_coder_charge(qieType, adc + 1u, capid, qieOffsets, qieSlopes) - q;
      else if (mantissa == 1u || mantissa == mantissaMask)
        return q - compute_coder_charge(qieType, adc - 1u, capid, qieOffsets, qieSlopes);
      else {
        auto const qup = compute_coder_charge(qieType, adc + 1u, capid, qieOffsets, qieSlopes);
        auto const qdown = compute_coder_charge(qieType, adc - 1u, capid, qieOffsets, qieSlopes);
        auto const upgain = qup - q;
        auto const downgain = q - qdown;
        auto const averagegain = (qup - qdown) / 2.f;
        if (std::abs(upgain - downgain) < 0.01f * averagegain)
          return averagegain;
        else {
          auto const q2up = compute_coder_charge(qieType, adc + 2u, capid, qieOffsets, qieSlopes);
          auto const q2down = compute_coder_charge(qieType, adc - 2u, capid, qieOffsets, qieSlopes);
          auto const upgain2 = q2up - qup;
          auto const downgain2 = qdown - q2down;
          if (std::abs(upgain2 - upgain) < std::abs(downgain2 - downgain))
            return upgain;
          else
            return downgain;
        }
      }
    }

    // TODO: remove what's not needed
    // originally from from RecoLocalCalo/HcalRecAlgos/src/PulseShapeFunctor.cc
    __forceinline__ __device__ float compute_pulse_shape_value(float const pulse_time,
                                                               int const sample,
                                                               int const shift,
                                                               float const* acc25nsVec,
                                                               float const* diff25nsItvlVec,
                                                               float const* accVarLenIdxMinusOneVec,
                                                               float const* diffVarItvlIdxMinusOneVec,
                                                               float const* accVarLenIdxZeroVec,
                                                               float const* diffVarItvlIdxZeroVec) {
      // constants
      constexpr float slew = 0.f;
      constexpr auto ns_per_bx = hcal::constants::nsPerBX;

      // FIXME: clean up all the rounding... this is coming from original cpu version
      float const i_start_float = -hcal::constants::iniTimeShift - pulse_time - slew > 0.f
                                      ? 0.f
                                      : std::abs(-hcal::constants::iniTimeShift - pulse_time - slew) + 1.f;
      int i_start = static_cast<int>(i_start_float);
      float offset_start = static_cast<float>(i_start) - hcal::constants::iniTimeShift - pulse_time - slew;
      // FIXME: do we need a check for nan???
#ifdef HCAL_MAHI_GPUDEBUG
      if (shift == 0)
        printf("i_start_float = %f i_start = %d offset_start = %f\n", i_start_float, i_start, offset_start);
#endif

      // boundary
      if (offset_start == 1.0f) {
        offset_start = 0.f;
        i_start -= 1;
      }

#ifdef HCAL_MAHI_GPUDEBUG
      if (shift == 0)
        printf("i_start_float = %f i_start = %d offset_start = %f\n", i_start_float, i_start, offset_start);
#endif

      int const bin_start = static_cast<int>(offset_start);
      auto const bin_start_up = static_cast<float>(bin_start) + 0.5f;
      int const bin_0_start = offset_start < bin_start_up ? bin_start - 1 : bin_start;
      int const its_start = i_start / ns_per_bx;
      int const distTo25ns_start = hcal::constants::nsPerBX - 1 - i_start % ns_per_bx;
      auto const factor = offset_start - static_cast<float>(bin_0_start) - 0.5;

#ifdef HCAL_MAHI_GPUDEBUG
      if (shift == 0) {
        printf("bin_start = %d bin_0_start = %d its_start = %d distTo25ns_start = %d factor = %f\n",
               bin_start,
               bin_0_start,
               its_start,
               distTo25ns_start,
               factor);
      }
#endif

      auto const sample_over10ts = sample + shift;
      float value = 0.0f;
      if (sample_over10ts == its_start) {
        value = bin_0_start == -1
                    ? accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
                    : accVarLenIdxZeroVec[distTo25ns_start] + factor * diffVarItvlIdxZeroVec[distTo25ns_start];
      } else if (sample_over10ts > its_start) {
        int const bin_idx = distTo25ns_start + 1 + (sample_over10ts - its_start - 1) * ns_per_bx + bin_0_start;
        value = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
      }
      return value;
    }

  }  // namespace reconstruction
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecProducers_src_KernelHelpers_h
