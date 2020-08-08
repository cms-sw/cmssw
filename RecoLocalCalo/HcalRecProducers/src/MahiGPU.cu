#include <Eigen/Dense>

#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"

// nvcc not able to parse this guy (whatever is inlcuded from it)....
//#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

#include "MahiGPU.h"

#ifdef HCAL_MAHI_GPUDEBUG
#define DETID_TO_DEBUG 1125647428
#endif

namespace hcal {
  namespace mahi {

    template <int NROWS, int NCOLS>
    using ColMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::ColMajor>;

    template <int NROWS, int NCOLS>
    using RowMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::RowMajor>;

    template <int SIZE, typename T = float>
    using ColumnVector = Eigen::Matrix<T, SIZE, 1>;

    template <int SIZE, typename T = float>
    using RowVector = Eigen::Matrix<T, 1, SIZE>;

    // FIXME remove duplication...
    // this is from PulesFunctor. nvcc was complaining... if included that header...
    //constexpr int maxSamples = 10;
    constexpr int maxPSshapeBin = 256;
    constexpr int nsPerBX = 25;
    constexpr float iniTimeShift = 92.5f;

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
      return did.ieta() > 0 ? value + maxDepthHB * IPHI_MAX * (did.ieta() - firstHBRing)
                            : value + maxDepthHB * IPHI_MAX * (did.ieta() + lastHBRing + nEtaHB);
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

    // Assume: same number of samples for HB and HE
    // TODO: add/validate restrict (will increase #registers in use by the kernel)
    __global__ void kernel_prep1d_sameNumberOfSamples(float* amplitudes,
                                                      float* noiseTerms,
                                                      float* outputEnergy,
                                                      float* outputChi2,
                                                      uint16_t const* dataf01HE,
                                                      uint16_t const* dataf5HB,
                                                      uint16_t const* dataf3HB,
                                                      uint32_t const* idsf01HE,
                                                      uint32_t const* idsf5HB,
                                                      uint32_t const* idsf3HB,
                                                      uint32_t const stridef01HE,
                                                      uint32_t const stridef5HB,
                                                      uint32_t const stridef3HB,
                                                      uint32_t const nchannelsf01HE,
                                                      uint32_t const nchannelsf5HB,
                                                      uint8_t const* npresamplesf5HB,
                                                      int8_t* soiSamples,
                                                      float* method0Energy,
                                                      float* method0Time,
                                                      uint32_t* outputdid,
                                                      uint32_t const nchannels,
                                                      uint32_t const* recoParam1Values,
                                                      uint32_t const* recoParam2Values,
                                                      float const* qieCoderOffsets,
                                                      float const* qieCoderSlopes,
                                                      int const* qieTypes,
                                                      float const* pedestalWidths,
                                                      float const* effectivePedestalWidths,
                                                      float const* pedestals,
                                                      float const* effectivePedestals,
                                                      bool const useEffectivePedestals,
                                                      int const* sipmTypeValues,
                                                      float const* fcByPEValues,
                                                      float const* parLin1Values,
                                                      float const* parLin2Values,
                                                      float const* parLin3Values,
                                                      float const* gainValues,
                                                      float const* respCorrectionValues,
                                                      int const maxDepthHB,
                                                      int const maxDepthHE,
                                                      int const maxPhiHE,
                                                      int const firstHBRing,
                                                      int const lastHBRing,
                                                      int const firstHERing,
                                                      int const lastHERing,
                                                      int const nEtaHB,
                                                      int const nEtaHE,
                                                      int const sipmQTSShift,
                                                      int const sipmQNTStoSum,
                                                      int const firstSampleShift,
                                                      uint32_t const offsetForHashes,
                                                      float const ts4Thresh,
                                                      int const startingSample) {
      // indices + runtime constants
      auto const sample = threadIdx.x + startingSample;
      auto const sampleWithinWindow = threadIdx.x;
      int32_t const nsamplesForCompute = blockDim.x;
      auto const lch = threadIdx.y;
      auto const gch = lch + blockDim.y * blockIdx.x;
      auto const nchannels_per_block = blockDim.y;
      auto const linearThPerBlock = threadIdx.x + threadIdx.y * blockDim.x;

      // remove
      if (gch >= nchannels)
        return;

      // initialize all output buffers
      if (sampleWithinWindow == 0) {
        outputdid[gch] = 0;
        method0Energy[gch] = 0;
        method0Time[gch] = 0;
        outputEnergy[gch] = 0;
        outputChi2[gch] = 0;
      }

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_SINGLECHANNEL
      if (gch > 0)
        return;
#endif
#endif

      // configure shared mem
      extern __shared__ char smem[];
      float* shrEnergyM0PerTS = reinterpret_cast<float*>(smem);
      float* shrChargeMinusPedestal = shrEnergyM0PerTS + nsamplesForCompute * nchannels_per_block;
      float* shrMethod0EnergyAccum = shrChargeMinusPedestal + nsamplesForCompute * nchannels_per_block;
      float* shrEnergyM0TotalAccum = shrMethod0EnergyAccum + nchannels_per_block;
      unsigned long long int* shrMethod0EnergySamplePair =
          reinterpret_cast<unsigned long long int*>(shrEnergyM0TotalAccum + nchannels_per_block);
      if (sampleWithinWindow == 0) {
        shrMethod0EnergyAccum[lch] = 0;
        shrMethod0EnergySamplePair[lch] = __float_as_uint(std::numeric_limits<float>::min());
        shrEnergyM0TotalAccum[lch] = 0;
      }

      // offset output
      auto* amplitudesForChannel = amplitudes + nsamplesForCompute * gch;
      auto* noiseTermsForChannel = noiseTerms + nsamplesForCompute * gch;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;

      // get event input quantities
      auto const stride = gch < nchannelsf01HE ? stridef01HE : (gch < nchannelsf015 ? stridef5HB : stridef3HB);
      auto const nsamples = gch < nchannelsf01HE ? compute_nsamples<Flavor01>(stride)
                                                 : (gch < nchannelsf015 ? compute_nsamples<Flavor5>(stride)
                                                                        : compute_nsamples<Flavor3>(stride));

#ifdef HCAL_MAHI_GPUDEBUG
      assert(nsamples == nsamplesForCompute || nsamples-startingSample==nsampelsForCompute);
#endif

      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      auto const did = HcalDetId{id};
      auto const adc =
          gch < nchannelsf01HE
              ? adc_for_sample<Flavor01>(dataf01HE + stride * gch, sample)
              : (gch < nchannelsf015 ? adc_for_sample<Flavor5>(dataf5HB + stride * (gch - nchannelsf01HE), sample)
                                     : adc_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));
      auto const capid =
          gch < nchannelsf01HE
              ? capid_for_sample<Flavor01>(dataf01HE + stride * gch, sample)
              : (gch < nchannelsf015 ? capid_for_sample<Flavor5>(dataf5HB + stride * (gch - nchannelsf01HE), sample)
                                     : capid_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

      // compute hash for this did
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) + offsetForHashes;

      // conditions based on the hash
      // FIXME: remove hardcoded values
      auto const qieType = qieTypes[hashedId] > 0 ? 1 : 0;  // 2 types at this point
      auto const* qieOffsets = qieCoderOffsets + hashedId * HcalQIECodersGPU::numValuesPerChannel;
      auto const* qieSlopes = qieCoderSlopes + hashedId * HcalQIECodersGPU::numValuesPerChannel;
      auto const* pedestalsForChannel = pedestals + hashedId * 4;
      auto const* pedestalWidthsForChannel = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                                 ? effectivePedestalWidths + hashedId * 4
                                                 : pedestalWidths + hashedId * 4;
      auto const* gains = gainValues + hashedId * 4;
      auto const gain = gains[capid];
      auto const gain0 = gains[0];
      auto const respCorrection = respCorrectionValues[hashedId];
      auto const pedestal = pedestalsForChannel[capid];
      auto const pedestalWidth = pedestalWidthsForChannel[capid];
      // if needed, only use effective pedestals for f01
      auto const pedestalToUseForMethod0 = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                               ? effectivePedestals[hashedId * 4 + capid]
                                               : pedestal;
      auto const sipmType = sipmTypeValues[hashedId];
      auto const fcByPE = fcByPEValues[hashedId];
      auto const recoParam1 = recoParam1Values[hashedId];
      auto const recoParam2 = recoParam2Values[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
      printf("qieType = %d qieOffset0 = %f qieOffset1 = %f qieSlope0 = %f qieSlope1 = %f\n",
             qieType,
             qieOffsets[0],
             qieOffsets[1],
             qieSlopes[0],
             qieSlopes[1]);
#endif

      // compute charge
      auto const charge = compute_coder_charge(qieType, adc, capid, qieOffsets, qieSlopes);

      shrChargeMinusPedestal[linearThPerBlock] = charge - pedestal;
      if (gch < nchannelsf01HE) {
        // NOTE: assume that soi is high only for a single guy!
        //   which must be the case. cpu version does not check for that
        //   if that is not the case, we will see that with cuda mmecheck
        auto const soibit = soibit_for_sample<Flavor01>(dataf01HE + stride * gch, sample);
        if (soibit == 1)
          soiSamples[gch] = sampleWithinWindow;
      } else if (gch >= nchannelsf015) {
        auto const soibit = soibit_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample);
        if (soibit == 1)
          soiSamples[gch] = sampleWithinWindow;
      }
      __syncthreads();
      int32_t const soi = gch < nchannelsf01HE
                              ? soiSamples[gch]
                              : (gch < nchannelsf015 ? npresamplesf5HB[gch - nchannelsf01HE] : soiSamples[gch]);
      //int32_t const soi = gch >= nchannelsf01HE
      //    ? npresamplesf5HB[gch - nchannelsf01HE]
      //    : soiSamples[gch];
      // this is here just to make things uniform...
      if (gch >= nchannelsf01HE && gch < nchannelsf015 && sampleWithinWindow == 0)
        soiSamples[gch] = npresamplesf5HB[gch - nchannelsf01HE];

      //
      // compute various quantities (raw charge and tdc stuff)
      // NOTE: this branch will be divergent only for a single warp that
      // sits on the boundary when flavor 01 channels end and flavor 5 start
      //
      float rawCharge;
#ifdef COMPUTE_TDC_TIME
      float tdcTime;
#endif  // COMPUTE_TDC_TIME
      auto const dfc = compute_diff_charge_gain(
          qieType, adc, capid, qieOffsets, qieSlopes, gch < nchannelsf01HE || gch >= nchannelsf015);
      if (gch >= nchannelsf01HE && gch < nchannelsf015) {
        // flavor 5
        rawCharge = charge;
#ifdef COMPUTE_TDC_TIME
        tdcTime = HcalSpecialTimes::UNKNOWN_T_NOTDC;
#endif  // COMPUTE_TDC_TIME
      } else {
        // flavor 0 or 1 or 3
        // conditions needed for sipms
        auto const parLin1 = parLin1Values[sipmType - 1];
        auto const parLin2 = parLin2Values[sipmType - 1];
        auto const parLin3 = parLin3Values[sipmType - 1];

        int const first = std::max(soi + sipmQTSShift, 0);
        int const last = std::min(soi + sipmQNTStoSum, nsamplesForCompute);
        float sipmq = 0.0f;
        for (auto ts = first; ts < last; ts++)
          sipmq += shrChargeMinusPedestal[threadIdx.y * nsamplesForCompute + ts];
        auto const effectivePixelsFired = sipmq / fcByPE;
        auto const factor = compute_reco_correction_factor(parLin1, parLin2, parLin3, effectivePixelsFired);
        rawCharge = (charge - pedestal) * factor + pedestal;
#ifdef COMPUTE_TDC_TIME
        if (gch < nchannelsf01HE)
          tdcTime = HcalSpecialTimes::getTDCTime(tdc_for_sample<Flavor01>(dataf01HE + stride * gch, sample));
        else if (gch >= nchannelsf015)
          tdcTime =
              HcalSpecialTimes::getTDCTime(tdc_for_sample<Flavor3>(dataf3HB + stride * (gch - nchannelsf015), sample));
#endif  // COMPUTE_TDC_TIME

#ifdef HCAL_MAHI_GPUDEBUG
        printf("first = %d last = %d sipmQ = %f factor = %f rawCharge = %f\n", first, last, sipmq, factor, rawCharge);
#endif
      }

      // compute method 0 quantities
      // TODO: need to apply containment
      // TODO: need to apply time slew
      // TODO: for < run 3, apply HBM legacy energy correction
      auto const nsamplesToAdd = recoParam1 < 10 ? recoParam2 : (recoParam1 >> 14) & 0xF;
      auto const startSampleTmp = soi + firstSampleShift;
      auto const startSample = startSampleTmp < 0 ? 0 : startSampleTmp;
      auto const endSample = startSample + nsamplesToAdd < nsamplesForCompute ? startSample + nsamplesToAdd : nsamplesForCompute;
      // NOTE: gain is a small number < 10^-3, multiply it last
      auto const energym0_per_ts = gain * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
      auto const energym0_per_ts_gain0 = gain0 * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
      // store to shared mem
      shrEnergyM0PerTS[lch * nsamplesForCompute + sampleWithinWindow] = energym0_per_ts;
      atomicAdd(&shrEnergyM0TotalAccum[lch], energym0_per_ts_gain0);

#ifdef HCAL_MAHI_GPUDEBUG
      printf(
          "id = %u sample = %d gch = %d hashedId = %u adc = %u capid = %u\n"
          "   charge = %f rawCharge = %f dfc = %f pedestal = %f\n"
          "   gain = %f respCorrection = %f energym0_per_ts = %f\n",
          id,
          sample,
          gch,
          hashedId,
          adc,
          capid,
          charge,
          rawCharge,
          dfc,
          pedestalToUseForMethod0,
          gain,
          respCorrection,
          energym0_per_ts);
      printf(
          "startSample = %d endSample = %d param1 = %u param2 = %u\n", startSample, endSample, recoParam1, recoParam2);
#endif

      if (sampleWithinWindow >= startSample && sampleWithinWindow < endSample) {
        atomicAdd(&shrMethod0EnergyAccum[lch], energym0_per_ts);
        // pack sample, energy as 64 bit value
        unsigned long long int old = shrMethod0EnergySamplePair[lch], assumed;
        unsigned long long int val =
            (static_cast<unsigned long long int>(sampleWithinWindow) << 32) + __float_as_uint(energym0_per_ts);
        do {
          assumed = old;
          // decode energy, sample values
          //int const current_sample = (assumed >> 32) & 0xffffffff;
          float const current_energy = __uint_as_float(assumed & 0xffffffff);
          if (energym0_per_ts > current_energy)
            old = atomicCAS(&shrMethod0EnergySamplePair[lch], assumed, val);
          else
            break;
        } while (assumed != old);
      }
      __syncthreads();

      // NOTE: must take soi, as values for that thread are used...
      if (sampleWithinWindow == soi) {
        auto const method0_energy = shrMethod0EnergyAccum[lch];
        auto const val = shrMethod0EnergySamplePair[lch];
        int const max_sample = (val >> 32) & 0xffffffff;
        float const max_energy = __uint_as_float(val & 0xffffffff);
        float const max_energy_1 =
            max_sample < nsamplesForCompute - 1 ? shrEnergyM0PerTS[lch * nsamplesForCompute + max_sample + 1] : 0.f;
        float const position = nsamplesToAdd < nsamplesForCompute ? max_sample - soi : max_sample;
        auto const sum = max_energy + max_energy_1;
        // FIXME: for full comparison with cpu method 0  timing,
        // need to correct by slew
        // requires an accumulator -> more shared mem -> omit here unless
        // really needed
        float const time =
            max_energy > 0.f && max_energy_1 > 0.f ? 25.f * (position + max_energy_1 / sum) : 25.f * position;

        // store method0 quantities to global mem
        outputdid[gch] = id;
        method0Energy[gch] = method0_energy;
        method0Time[gch] = time;

#ifdef HCAL_MAHI_GPUDEBUG
        printf("tsTOT = %f tstrig = %f ts4Thresh = %f\n", shrEnergyM0TotalAccum[lch], energym0_per_ts_gain0, ts4Thresh);
#endif

        // check as in cpu version if mahi is not needed
        // FIXME: KNOWN ISSUE: observed a problem when rawCharge and pedestal
        // are basically equal and generate -0.00000...
        // needs to be treated properly
        if (!(shrEnergyM0TotalAccum[lch] > 0 && energym0_per_ts_gain0 >= ts4Thresh)) {
          // do not need to run mahi minimization
          //outputEnergy[gch] = 0; energy already inited to 0
          outputChi2[gch] = -9999.f;
        }

#ifdef HCAL_MAHI_GPUDEBUG
        printf("method0_energy = %f max_sample = %d max_energy = %f time = %f\n",
               method0_energy,
               max_sample,
               max_energy,
               time);
#endif
      }

      //
      // preparations for mahi fit
      //
      auto const amplitude = rawCharge - pedestalToUseForMethod0;
      auto const noiseADC = (1. / std::sqrt(12)) * dfc;
      auto const noisePhoto = amplitude > pedestalWidth ? std::sqrt(amplitude * fcByPE) : 0.f;
      auto const noiseTerm = noiseADC * noiseADC + noisePhoto * noisePhoto + pedestalWidth * pedestalWidth;

#ifdef HCAL_MAHI_GPUDEBUG
      printf(
          "charrge(%d) = %f pedestal(%d) = %f dfc(%d) = %f pedestalWidth(%d) = %f noiseADC(%d) = %f noisPhoto(%d) = "
          "%f\n",
          sample,
          rawCharge,
          sample,
          pedestalToUseForMethod0,
          sample,
          dfc,
          sample,
          pedestalWidth,
          sample,
          noiseADC,
          sample,
          noisePhoto);
#endif

      // store to global memory
      amplitudesForChannel[sampleWithinWindow] = amplitude;
      noiseTermsForChannel[sampleWithinWindow] = noiseTerm;
    }

    // TODO: remove what's not needed
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
      constexpr float pulse_height = 1.0f;
      constexpr float slew = 0.f;
      constexpr auto ns_per_bx = nsPerBX;
      //constexpr auto num_ns = nsPerBX * maxSamples;
      //constexpr auto num_bx = num_ns / ns_per_bx;

      // FIXME: clean up all the rounding... this is coming from original cpu version
      float const i_start_float =
          -iniTimeShift - pulse_time - slew > 0.f ? 0.f : std::abs(-iniTimeShift - pulse_time - slew) + 1.f;
      int i_start = static_cast<int>(i_start_float);
      float offset_start = static_cast<float>(i_start) - iniTimeShift - pulse_time - slew;
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
      int const distTo25ns_start = nsPerBX - 1 - i_start % ns_per_bx;
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
      value *= pulse_height;
      return value;
    }

    // TODO: need to add an array of offsets for pulses (a la activeBXs...)
    // Assume for now 8 pulses
    __global__ void kernel_prep_pulseMatrices_sameNumberOfSamples(float* pulseMatrices,
                                                                  float* pulseMatricesM,
                                                                  float* pulseMatricesP,
                                                                  int const* pulseOffsets,
                                                                  float const* amplitudes,
                                                                  uint32_t const* idsf01HE,
                                                                  uint32_t const* idsf5HB,
                                                                  uint32_t const* idsf3HB,
                                                                  uint32_t const nchannelsf01HE,
                                                                  uint32_t const nchannelsf5HB,
                                                                  uint32_t const nchannelsTotal,
                                                                  int8_t const* soiSamples,
                                                                  uint32_t const* recoPulseShapeIds,
                                                                  float const* acc25nsVecValues,
                                                                  float const* diff25nsItvlVecValues,
                                                                  float const* accVarLenIdxMinusOneVecValues,
                                                                  float const* diffVarItvlIdxMinusOneVecValues,
                                                                  float const* accVarLenIdxZeroVecValues,
                                                                  float const* diffVarItvlIdxZeroVecValues,
                                                                  float const meanTime,
                                                                  float const timeSigmaSiPM,
                                                                  float const timeSigmaHPD,
                                                                  int const maxDepthHB,
                                                                  int const maxDepthHE,
                                                                  int const maxPhiHE,
                                                                  int const firstHBRing,
                                                                  int const lastHBRing,
                                                                  int const firstHERing,
                                                                  int const lastHERing,
                                                                  int const nEtaHB,
                                                                  int const nEtaHE,
                                                                  uint32_t const offsetForHashes,
                                                                  bool const applyTimeSlew,
                                                                  float const tzeroTimeSlew,
                                                                  float const slopeTimeSlew,
                                                                  float const tmaxTimeSlew) {
      // indices
      auto const ipulse = threadIdx.y;
      auto const npulses = blockDim.y;
      auto const sample = threadIdx.x;
      auto const nsamples = blockDim.x;
      auto const lch = threadIdx.z;
      auto const gch = lch + blockIdx.x * blockDim.z;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;

      if (gch >= nchannelsTotal)
        return;

      // conditions
      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      //auto const id = gch >= nchannelsf01HE
      //    ? idsf5HB[gch - nchannelsf01HE]
      //    : idsf01HE[gch];
      auto const deltaT = gch >= nchannelsf01HE && gch < nchannelsf015 ? timeSigmaHPD : timeSigmaSiPM;
      auto const did = DetId{id};
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) + offsetForHashes;
      auto const recoPulseShapeId = recoPulseShapeIds[hashedId];
      auto const* acc25nsVec = acc25nsVecValues + recoPulseShapeId * maxPSshapeBin;
      auto const* diff25nsItvlVec = diff25nsItvlVecValues + recoPulseShapeId * maxPSshapeBin;
      auto const* accVarLenIdxMinusOneVec = accVarLenIdxMinusOneVecValues + recoPulseShapeId * nsPerBX;
      auto const* diffVarItvlIdxMinusOneVec = diffVarItvlIdxMinusOneVecValues + recoPulseShapeId * nsPerBX;
      auto const* accVarLenIdxZeroVec = accVarLenIdxZeroVecValues + recoPulseShapeId * nsPerBX;
      auto const* diffVarItvlIdxZeroVec = diffVarItvlIdxZeroVecValues + recoPulseShapeId * nsPerBX;

      // offset output arrays
      auto* pulseMatrix = pulseMatrices + nsamples * npulses * gch;
      auto* pulseMatrixM = pulseMatricesM + nsamples * npulses * gch;
      auto* pulseMatrixP = pulseMatricesP + nsamples * npulses * gch;

      // amplitude per ipulse
      int const soi = soiSamples[gch];
      int const pulseOffset = pulseOffsets[ipulse];
      auto const amplitude = amplitudes[gch * nsamples + pulseOffset + soi];

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

#ifdef HCAL_MAHI_GPUDEBUG
      if (sample == 0 && ipulse == 0) {
        for (int i = 0; i < 8; i++)
          printf("amplitude(%d) = %f\n", i, amplitudes[gch * nsamples + i]);
        printf("acc25nsVec and diff25nsItvlVec for recoPulseShapeId = %u\n", recoPulseShapeId);
        for (int i = 0; i < 256; i++) {
          printf("acc25nsVec(%d) = %f diff25nsItvlVec(%d) = %f\n", i, acc25nsVec[i], i, diff25nsItvlVec[i]);
        }
        printf("accVarLenIdxZEROVec and accVarLenIdxMinusOneVec\n");
        for (int i = 0; i < 25; i++) {
          printf("accVarLenIdxZEROVec(%d) = %f accVarLenIdxMinusOneVec(%d) = %f\n",
                 i,
                 accVarLenIdxZeroVec[i],
                 i,
                 accVarLenIdxMinusOneVec[i]);
        }
        printf("diffVarItvlIdxZEROVec and diffVarItvlIdxMinusOneVec\n");
        for (int i = 0; i < 25; i++) {
          printf("diffVarItvlIdxZEROVec(%d) = %f diffVarItvlIdxMinusOneVec(%d) = %f\n",
                 i,
                 diffVarItvlIdxZeroVec[i],
                 i,
                 diffVarItvlIdxMinusOneVec[i]);
        }
      }
#endif

      auto t0 = meanTime;
      if (applyTimeSlew) {
        if (amplitude <= 1.0f)
          t0 += compute_time_slew_delay(1.0, tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew);
        else
          t0 += compute_time_slew_delay(amplitude, tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew);
      }
      auto const t0m = -deltaT + t0;
      auto const t0p = deltaT + t0;

#ifdef HCAL_MAHI_GPUDEBUG
      if (sample == 0 && ipulse == 0) {
        printf("time values: %f %f %f\n", t0, t0m, t0p);
      }

      if (sample == 0 && ipulse == 0) {
        for (int i = 0; i < 10; i++) {
          auto const value = compute_pulse_shape_value(t0,
                                                       i,
                                                       0,
                                                       acc25nsVec,
                                                       diff25nsItvlVec,
                                                       accVarLenIdxMinusOneVec,
                                                       diffVarItvlIdxMinusOneVec,
                                                       accVarLenIdxZeroVec,
                                                       diffVarItvlIdxZeroVec);
          printf("pulse(%d) = %f\n", i, value);
        }
        printf("\n");
        for (int i = 0; i < 10; i++) {
          auto const value = compute_pulse_shape_value(t0p,
                                                       i,
                                                       0,
                                                       acc25nsVec,
                                                       diff25nsItvlVec,
                                                       accVarLenIdxMinusOneVec,
                                                       diffVarItvlIdxMinusOneVec,
                                                       accVarLenIdxZeroVec,
                                                       diffVarItvlIdxZeroVec);
          printf("pulseP(%d) = %f\n", i, value);
        }
        printf("\n");
        for (int i = 0; i < 10; i++) {
          auto const value = compute_pulse_shape_value(t0m,
                                                       i,
                                                       0,
                                                       acc25nsVec,
                                                       diff25nsItvlVec,
                                                       accVarLenIdxMinusOneVec,
                                                       diffVarItvlIdxMinusOneVec,
                                                       accVarLenIdxZeroVec,
                                                       diffVarItvlIdxZeroVec);
          printf("pulseM(%d) = %f\n", i, value);
        }
      }
#endif

      // FIXME: shift should be treated properly,
      // here assume 8 time slices and 8 samples
      auto const shift = 4 - soi;  // as in cpu version!

      // auto const offset = ipulse - soi;
      // auto const idx = sample - offset;
      int32_t const idx = sample - pulseOffset;
      auto const value = idx >= 0 && idx < nsamples ? compute_pulse_shape_value(t0,
                                                                                idx,
                                                                                shift,
                                                                                acc25nsVec,
                                                                                diff25nsItvlVec,
                                                                                accVarLenIdxMinusOneVec,
                                                                                diffVarItvlIdxMinusOneVec,
                                                                                accVarLenIdxZeroVec,
                                                                                diffVarItvlIdxZeroVec)
                                                    : 0;
      auto const value_t0m = idx >= 0 && idx < nsamples ? compute_pulse_shape_value(t0m,
                                                                                    idx,
                                                                                    shift,
                                                                                    acc25nsVec,
                                                                                    diff25nsItvlVec,
                                                                                    accVarLenIdxMinusOneVec,
                                                                                    diffVarItvlIdxMinusOneVec,
                                                                                    accVarLenIdxZeroVec,
                                                                                    diffVarItvlIdxZeroVec)
                                                        : 0;
      auto const value_t0p = idx >= 0 && idx < nsamples ? compute_pulse_shape_value(t0p,
                                                                                    idx,
                                                                                    shift,
                                                                                    acc25nsVec,
                                                                                    diff25nsItvlVec,
                                                                                    accVarLenIdxMinusOneVec,
                                                                                    diffVarItvlIdxMinusOneVec,
                                                                                    accVarLenIdxZeroVec,
                                                                                    diffVarItvlIdxZeroVec)
                                                        : 0;

      // store to global
      pulseMatrix[ipulse * nsamples + sample] = value;
      ;
      pulseMatrixM[ipulse * nsamples + sample] = value_t0m;
      pulseMatrixP[ipulse * nsamples + sample] = value_t0p;
    }

    // FIXME: provide specialization for Row Major layout
    template <typename T, int Stride, int Order = Eigen::ColMajor>
    struct MapSymM {
      using type = T;
      using base_type = typename std::remove_const<type>::type;

      static constexpr int total = Stride * (Stride + 1) / 2;
      static constexpr int stride = Stride;
      T* data;

      __forceinline__ __device__ MapSymM(T* data) : data{data} {}

      __forceinline__ __device__ T const& operator()(int const row, int const col) const {
        auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
        auto const index = total - tmp + row - col;
        return data[index];
      }

      template <typename U = T>
      __forceinline__ __device__ typename std::enable_if<std::is_same<base_type, U>::value, base_type>::type&
      operator()(int const row, int const col) {
        auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
        auto const index = total - tmp + row - col;
        return data[index];
      }
    };

    // simple/trivial cholesky decomposition impl
    template <typename MatrixType1, typename MatrixType2>
    __forceinline__ __device__ void compute_decomposition_unrolled(MatrixType1& L, MatrixType2 const& M) {
      auto const sqrtm_0_0 = std::sqrt(M(0, 0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;

#pragma unroll
      for (int i = 1; i < MatrixType1::stride; i++) {
        T sumsq{0};
        for (int j = 0; j < i; j++) {
          T sumsq2{0};
          auto const m_i_j = M(i, j);
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
        }

        auto const l_i_i = std::sqrt(M(i, i) - sumsq);
        L(i, i) = l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2>
    __forceinline__ __device__ void compute_decomposition(MatrixType1& L, MatrixType2 const& M, int const N) {
      auto const sqrtm_0_0 = std::sqrt(M(0, 0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;

      for (int i = 1; i < N; i++) {
        T sumsq{0};
        for (int j = 0; j < i; j++) {
          T sumsq2{0};
          auto const m_i_j = M(i, j);
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
        }

        auto const l_i_i = std::sqrt(M(i, i) - sumsq);
        L(i, i) = l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2, typename VectorType>
    __forceinline__ __device__ void compute_decomposition_forwardsubst_with_offsets(
        MatrixType1& L,
        MatrixType2 const& M,
        float b[MatrixType1::stride],
        VectorType const& Atb,
        int const N,
        ColumnVector<MatrixType1::stride, int> const& pulseOffsets) {
      auto const real_0 = pulseOffsets(0);
      auto const sqrtm_0_0 = std::sqrt(M(real_0, real_0));
      L(0, 0) = sqrtm_0_0;
      using T = typename MatrixType1::base_type;
      b[0] = Atb(real_0) / sqrtm_0_0;

      for (int i = 1; i < N; i++) {
        auto const i_real = pulseOffsets(i);
        T sumsq{0};
        T total = 0;
        auto const atb = Atb(i_real);
        for (int j = 0; j < i; j++) {
          auto const j_real = pulseOffsets(j);
          T sumsq2{0};
          auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));
          for (int k = 0; k < j; ++k)
            sumsq2 += L(i, k) * L(j, k);

          auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
          L(i, j) = value_i_j;

          sumsq += value_i_j * value_i_j;
          total += value_i_j * b[j];
        }

        auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
        L(i, i) = l_i_i;
        b[i] = (atb - total) / l_i_i;
      }
    }

    template <typename MatrixType1, typename MatrixType2, typename VectorType>
    __forceinline__ __device__ void update_decomposition_forwardsubst_with_offsets(
        MatrixType1& L,
        MatrixType2 const& M,
        float b[MatrixType1::stride],
        VectorType const& Atb,
        int const N,
        ColumnVector<MatrixType1::stride, int> const& pulseOffsets) {
      using T = typename MatrixType1::base_type;
      auto const i = N - 1;
      auto const i_real = pulseOffsets(i);
      T sumsq{0};
      T total = 0;
      for (int j = 0; j < i; j++) {
        auto const j_real = pulseOffsets(j);
        T sumsq2{0};
        auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));
        for (int k = 0; k < j; ++k)
          sumsq2 += L(i, k) * L(j, k);

        auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
        L(i, j) = value_i_j;
        sumsq += value_i_j * value_i_j;

        total += value_i_j * b[j];
      }

      auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
      L(i, i) = l_i_i;
      b[i] = (Atb(i_real) - total) / l_i_i;
    }

    template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
    __device__ void solve_forward_subst_matrix(MatrixType1& A,
                                               MatrixType2 const& pulseMatrixView,
                                               MatrixType3 const& matrixL) {
      // FIXME: this assumes pulses are on columns and samples on rows
      constexpr auto NPULSES = MatrixType2::ColsAtCompileTime;
      constexpr auto NSAMPLES = MatrixType2::RowsAtCompileTime;

#pragma unroll
      for (int icol = 0; icol < NPULSES; icol++) {
        float reg_b[NSAMPLES];
        float reg_L[NSAMPLES];

// preload a column and load column 0 of cholesky
#pragma unroll
        for (int i = 0; i < NSAMPLES; i++) {
          reg_b[i] = __ldg(&pulseMatrixView.coeffRef(i, icol));
          reg_L[i] = matrixL(i, 0);
        }

        // compute x0 and store it
        auto x_prev = reg_b[0] / reg_L[0];
        A(0, icol) = x_prev;

// iterate
#pragma unroll
        for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
          for (int counter = iL; counter < NSAMPLES; counter++)
            reg_b[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
          for (int counter = iL; counter < NSAMPLES; counter++)
            reg_L[counter] = matrixL(counter, iL);

          // compute the next x for M(iL, icol)
          x_prev = reg_b[iL] / reg_L[iL];

          // store the result value
          A(iL, icol) = x_prev;
        }
      }
    }

    template <typename MatrixType1, typename MatrixType2>
    __device__ void solve_forward_subst_vector(float reg_b[MatrixType1::RowsAtCompileTime],
                                               MatrixType1 inputAmplitudesView,
                                               MatrixType2 matrixL) {
      constexpr auto NSAMPLES = MatrixType1::RowsAtCompileTime;

      float reg_b_tmp[NSAMPLES];
      float reg_L[NSAMPLES];

// preload a column and load column 0 of cholesky
#pragma unroll
      for (int i = 0; i < NSAMPLES; i++) {
        reg_b_tmp[i] = inputAmplitudesView(i);
        reg_L[i] = matrixL(i, 0);
      }

      // compute x0 and store it
      auto x_prev = reg_b_tmp[0] / reg_L[0];
      reg_b[0] = x_prev;

// iterate
#pragma unroll
      for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
        for (int counter = iL; counter < NSAMPLES; counter++)
          reg_b_tmp[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
        for (int counter = iL; counter < NSAMPLES; counter++)
          reg_L[counter] = matrixL(counter, iL);

        // compute the next x for M(iL, icol)
        x_prev = reg_b_tmp[iL] / reg_L[iL];

        // store the result value
        reg_b[iL] = x_prev;
      }
    }

    // TODO: add active bxs
    template <typename MatrixType, typename VectorType>
    __device__ void fnnls(MatrixType const& AtA,
                          VectorType const& Atb,
                          VectorType& solution,
                          int& npassive,
                          ColumnVector<VectorType::RowsAtCompileTime, int>& pulseOffsets,
                          MapSymM<float, VectorType::RowsAtCompileTime>& matrixL,
                          double const eps,
                          int const maxIterations) {
      // constants
      constexpr auto NPULSES = VectorType::RowsAtCompileTime;

      // to keep track of where to terminate if converged
      Eigen::Index w_max_idx_prev = 0;
      float w_max_prev = 0;
      auto eps_to_use = eps;
      bool recompute = false;

      // used throughout
      VectorType s;
      float reg_b[NPULSES];
      //float matrixLStorage[MapSymM<float, NPULSES>::total];
      //MapSymM<float, NPULSES> matrixL{matrixLStorage};

      int iter = 0;
      while (true) {
        if (iter > 0 || npassive == 0) {
          auto const nactive = NPULSES - npassive;
          // exit if there are no more pulses to constrain
          if (nactive == 0)
            break;

          // compute the gradient
          //w.tail(nactive) = Atb.tail(nactive) - (AtA * solution).tail(nactive);
          Eigen::Index w_max_idx;
          float w_max = -std::numeric_limits<float>::max();
          for (int icol = npassive; icol < NPULSES; icol++) {
            auto const icol_real = pulseOffsets(icol);
            auto const atb = Atb(icol_real);
            float sum = 0;
#pragma unroll
            for (int counter = 0; counter < NPULSES; counter++)
              sum += counter > icol_real ? AtA(counter, icol_real) * solution(counter)
                                         : AtA(icol_real, counter) * solution(counter);

            auto const w = atb - sum;
            if (w > w_max) {
              w_max = w;
              w_max_idx = icol - npassive;
            }
          }

          // check for convergence
          if (w_max < eps_to_use || w_max_idx == w_max_idx_prev && w_max == w_max_prev)
            break;

          if (iter >= maxIterations)
            break;

          w_max_prev = w_max;
          w_max_idx_prev = w_max_idx;

          // move index to the right part of the vector
          w_max_idx += npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(w_max_idx));
          ++npassive;
        }

        // inner loop
        while (true) {
          if (npassive == 0)
            break;

          //s.head(npassive)
          //auto const& matrixL =
          //    AtA.topLeftCorner(npassive, npassive)
          //        .llt().matrixL();
          //.solve(Atb.head(npassive));
          if (recompute || iter == 0)
            compute_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);
          else
            update_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive, pulseOffsets);

          // run backward substituion
          s(npassive - 1) = reg_b[npassive - 1] / matrixL(npassive - 1, npassive - 1);
          for (int i = npassive - 2; i >= 0; --i) {
            float total = 0;
            for (int j = i + 1; j < npassive; j++)
              total += matrixL(j, i) * s(j);

            s(i) = (reg_b[i] - total) / matrixL(i, i);
          }

          // done if solution values are all positive
          if (s.head(npassive).minCoeff() > 0.f) {
            for (int i = 0; i < npassive; i++) {
              auto const i_real = pulseOffsets(i);
              solution(i_real) = s(i);
            }
            //solution.head(npassive) = s.head(npassive);
            recompute = false;
            break;
          }

          // there were negative values -> have to recompute the whole decomp
          recompute = true;

          auto alpha = std::numeric_limits<float>::max();
          Eigen::Index alpha_idx = 0, alpha_idx_real = 0;
          for (int i = 0; i < npassive; i++) {
            if (s[i] <= 0.) {
              auto const i_real = pulseOffsets(i);
              auto const ratio = solution[i_real] / (solution[i_real] - s[i]);
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
                alpha_idx_real = i_real;
              }
            }
          }

          // upadte solution
          for (int i = 0; i < npassive; i++) {
            auto const i_real = pulseOffsets(i);
            solution(i_real) += alpha * (s(i) - solution(i_real));
          }
          //solution.head(npassive) += alpha *
          //    (s.head(npassive) - solution.head(npassive));
          solution[alpha_idx_real] = 0;
          --npassive;

          Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(alpha_idx));
        }

        // as in cpu
        ++iter;
        if (iter % 10 == 0)
          eps_to_use *= 10;
      }
    }

    template <int NSAMPLES, int NPULSES>
    __forceinline__ __device__ void update_covariance(
        ColumnVector<NPULSES> const& resultAmplitudesVector,
        MapSymM<float, NSAMPLES>& covarianceMatrix,
        Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrix,
        Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixM,
        Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixP) {
#pragma unroll
      for (int ipulse = 0; ipulse < NPULSES; ipulse++) {
        auto const resultAmplitude = resultAmplitudesVector(ipulse);
        if (resultAmplitude == 0)
          continue;

#ifdef HCAL_MAHI_GPUDEBUG
        printf("pulse cov array for ibx = %d and offset %d\n", ipulse, offset);
#endif

        // preload a column
        float pmcol[NSAMPLES], pmpcol[NSAMPLES], pmmcol[NSAMPLES];
#pragma unroll
        for (int counter = 0; counter < NSAMPLES; counter++) {
          pmcol[counter] = __ldg(&pulseMatrix.coeffRef(counter, ipulse));
          pmpcol[counter] = __ldg(&pulseMatrixP.coeffRef(counter, ipulse));
          pmmcol[counter] = __ldg(&pulseMatrixM.coeffRef(counter, ipulse));
        }

        auto const ampl2 = resultAmplitude * resultAmplitude;
#pragma unroll
        for (int col = 0; col < NSAMPLES; col++) {
          auto const valueP_col = pmpcol[col];
          auto const valueM_col = pmmcol[col];
          auto const value_col = pmcol[col];
          auto const tmppcol = valueP_col - value_col;
          auto const tmpmcol = valueM_col - value_col;

          // diagonal
          auto tmp_value = 0.5 * (tmppcol * tmppcol + tmpmcol * tmpmcol);
          covarianceMatrix(col, col) += ampl2 * tmp_value;

// FIXME: understand if this actually gets unrolled
#pragma unroll
          for (int row = col + 1; row < NSAMPLES; row++) {
            float const valueP_row = pmpcol[row];  //pulseMatrixP(j, ipulseReal);
            float const value_row = pmcol[row];    //pulseMatrix(j, ipulseReal);
            float const valueM_row = pmmcol[row];  //pulseMatrixM(j, ipulseReal);

            float tmpprow = valueP_row - value_row;
            float tmpmrow = valueM_row - value_row;

            auto const covValue = 0.5 * (tmppcol * tmpprow + tmpmcol * tmpmrow);

            covarianceMatrix(row, col) += ampl2 * covValue;
          }
        }
      }
    }

    template <int NSAMPLES, int NPULSES>
    __global__ void kernel_minimize(float* outputEnergy,
                                    float* outputChi2,
                                    float const* __restrict__ inputAmplitudes,
                                    float const* __restrict__ pulseMatrices,
                                    float const* __restrict__ pulseMatricesM,
                                    float const* __restrict__ pulseMatricesP,
                                    int const* __restrict__ pulseOffsetValues,
                                    float const* __restrict__ noiseTerms,
                                    int8_t const* __restrict__ soiSamples,
                                    float const* __restrict__ pedestalWidths,
                                    float const* __restrict__ effectivePedestalWidths,
                                    bool const useEffectivePedestals,
                                    uint32_t const* __restrict__ idsf01HE,
                                    uint32_t const* __restrict__ idsf5HB,
                                    uint32_t const* __restrict__ idsf3HB,
                                    float const* __restrict__ gainValues,
                                    float const* __restrict__ respCorrectionValues,
                                    uint32_t const nchannelsf01HE,
                                    uint32_t const nchannelsf5HB,
                                    uint32_t const nchannelsTotal,
                                    uint32_t const offsetForHashes,
                                    int const maxDepthHB,
                                    int const maxDepthHE,
                                    int const maxPhiHE,
                                    int const firstHBRing,
                                    int const lastHBRing,
                                    int const firstHERing,
                                    int const lastHERing,
                                    int const nEtaHB,
                                    int const nEtaHE) {
      // can be relaxed if needed - minor updates are needed in that case!
      static_assert(NPULSES == NSAMPLES);

      // indices
      auto const gch = threadIdx.x + blockIdx.x * blockDim.x;
      auto const nchannelsf015 = nchannelsf01HE + nchannelsf5HB;
      if (gch >= nchannelsTotal)
        return;

      // if chi2 is set to -9999 do not run minimization
      if (outputChi2[gch] == -9999.f)
        return;

      // configure shared mem
      extern __shared__ char shrmem[];
      float* shrMatrixLFnnlsStorage = reinterpret_cast<float*>(shrmem) + MapSymM<float, NPULSES>::total * threadIdx.x;
      float* shrAtAStorage =
          reinterpret_cast<float*>(shrmem) + MapSymM<float, NPULSES>::total * (threadIdx.x + blockDim.x);

      // conditions for pedestal widths
      auto const id = gch < nchannelsf01HE
                          ? idsf01HE[gch]
                          : (gch < nchannelsf015 ? idsf5HB[gch - nchannelsf01HE] : idsf3HB[gch - nchannelsf015]);
      //auto const id = gch >= nchannelsf01HE
      //    ? idsf5HB[gch - nchannelsf01HE]
      //    : idsf01HE[gch];
      auto const did = DetId{id};
      auto const hashedId =
          did.subdetId() == HcalBarrel
              ? did2linearIndexHB(id, maxDepthHB, firstHBRing, lastHBRing, nEtaHB)
              : did2linearIndexHE(id, maxDepthHE, maxPhiHE, firstHERing, lastHERing, nEtaHE) + offsetForHashes;

      auto const* pedestalWidthsForChannel = useEffectivePedestals && (gch < nchannelsf01HE || gch >= nchannelsf015)
                                                 ? effectivePedestalWidths + hashedId * 4
                                                 : pedestalWidths + hashedId * 4;
      auto const averagePedestalWidth2 = 0.25 * (pedestalWidthsForChannel[0] * pedestalWidthsForChannel[0] +
                                                 pedestalWidthsForChannel[1] * pedestalWidthsForChannel[1] +
                                                 pedestalWidthsForChannel[2] * pedestalWidthsForChannel[2] +
                                                 pedestalWidthsForChannel[3] * pedestalWidthsForChannel[3]);
      auto const* gains = gainValues + hashedId * 4;
      // FIXME on cpu ts 0 capid was used - does it make any difference
      auto const gain = gains[0];
      auto const respCorrection = respCorrectionValues[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
      if (id != DETID_TO_DEBUG)
        return;
#endif
#endif

      /*
      // TODO: provide this properly
      int const soi = soiSamples[gch];
      */
      constexpr float deltaChi2Threashold = 1e-3;

      ColumnVector<NPULSES, int> pulseOffsets;
#pragma unroll
      for (int i = 0; i < NPULSES; ++i)
        pulseOffsets(i) = i;
      //        pulseOffsets(i) = pulseOffsetValues[i] - pulseOffsetValues[0];

      // output amplitudes/weights
      ColumnVector<NPULSES> resultAmplitudesVector = ColumnVector<NPULSES>::Zero();

      // map views
      Eigen::Map<const ColumnVector<NSAMPLES>> inputAmplitudesView{inputAmplitudes + gch * NSAMPLES};
      Eigen::Map<const ColumnVector<NSAMPLES>> noiseTermsView{noiseTerms + gch * NSAMPLES};
      Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixMView{pulseMatricesM +
                                                                              gch * NSAMPLES * NPULSES};
      Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixPView{pulseMatricesP +
                                                                              gch * NSAMPLES * NPULSES};
      Eigen::Map<const ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixView{pulseMatrices + gch * NSAMPLES * NPULSES};

#ifdef HCAL_MAHI_GPUDEBUG
      for (int i = 0; i < NSAMPLES; i++)
        printf("inputValues(%d) = %f noiseTerms(%d) = %f\n", i, inputAmplitudesView(i), i, noiseTermsView(i));
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixView(i, j));
        printf("\n");
      }
      printf("\n");
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixMView(i, j));
        printf("\n");
      }
      printf("\n");
      for (int i = 0; i < NSAMPLES; i++) {
        for (int j = 0; j < NPULSES; j++)
          printf("%f ", glbPulseMatrixPView(i, j));
        printf("\n");
      }
#endif

      int npassive = 0;
      float chi2 = 0, previous_chi2 = 0.f, chi2_2itersback = 0.f;
      // TOOD: provide constants from configuration
      for (int iter = 1; iter < 50; iter++) {
        //float covarianceMatrixStorage[MapSymM<float, NSAMPLES>::total];
        // NOTE: only works when NSAMPLES == NPULSES
        // if does not hold -> slightly rearrange shared mem to still reuse
        // shared memory
        float* covarianceMatrixStorage = shrMatrixLFnnlsStorage;
        MapSymM<float, NSAMPLES> covarianceMatrix{covarianceMatrixStorage};
#pragma unroll
        for (int counter = 0; counter < MapSymM<float, NSAMPLES>::total; counter++)
          covarianceMatrixStorage[counter] = averagePedestalWidth2;
#pragma unroll
        for (int counter = 0; counter < MapSymM<float, NSAMPLES>::stride; counter++)
          covarianceMatrix(counter, counter) += __ldg(&noiseTermsView.coeffRef(counter));

        // update covariance matrix
        update_covariance(
            resultAmplitudesVector, covarianceMatrix, glbPulseMatrixView, glbPulseMatrixMView, glbPulseMatrixPView);

#ifdef HCAL_MAHI_GPUDEBUG
        printf("covariance matrix\n");
        for (int i = 0; i < 8; i++) {
          for (int j = 0; j < 8; j++)
            printf("%f ", covarianceMatrix(i, j));
          printf("\n");
        }
#endif

        // compute Cholesky Decomposition L matrix
        //matrixDecomposition.compute(covarianceMatrix);
        //auto const& matrixL = matrixDecomposition.matrixL();
        float matrixLStorage[MapSymM<float, NSAMPLES>::total];
        MapSymM<float, NSAMPLES> matrixL{matrixLStorage};
        compute_decomposition_unrolled(matrixL, covarianceMatrix);

        //
        // replace eigen
        //
        //auto const& A = matrixDecomposition
        //    .matrixL()
        //    .solve(pulseMatrixView);
        ColMajorMatrix<NSAMPLES, NPULSES> A;
        solve_forward_subst_matrix(A, glbPulseMatrixView, matrixL);

        //
        // remove eigen
        //
        //auto const& b = matrixL
        //   .solve(inputAmplitudesView);
        //
        float reg_b[NSAMPLES];
        solve_forward_subst_vector(reg_b, inputAmplitudesView, matrixL);

        // TODO: we do not really need to change these matrcies
        // will be fixed in the optimized version
        //ColMajorMatrix<NPULSES, NPULSES> AtA = A.transpose() * A;
        //ColumnVector<NPULSES> Atb = A.transpose() * b;
        //ColMajorMatrix<NPULSES, NPULSES> AtA;
        //float AtAStorage[MapSymM<float, NPULSES>::total];
        MapSymM<float, NPULSES> AtA{shrAtAStorage};
        ColumnVector<NPULSES> Atb;
#pragma unroll
        for (int icol = 0; icol < NPULSES; icol++) {
          float reg_ai[NSAMPLES];

// load column icol
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            reg_ai[counter] = A(counter, icol);

          // compute diagonal
          float sum = 0.f;
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            sum += reg_ai[counter] * reg_ai[counter];

          // store
          AtA(icol, icol) = sum;

// go thru the other columns
#pragma unroll
          for (int j = icol + 1; j < NPULSES; j++) {
            // load column j
            float reg_aj[NSAMPLES];
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              reg_aj[counter] = A(counter, j);

            // accum
            float sum = 0.f;
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              sum += reg_aj[counter] * reg_ai[counter];

            // store
            //AtA(icol, j) = sum;
            AtA(j, icol) = sum;
          }

          // Atb accum
          float sum_atb = 0;
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            sum_atb += reg_ai[counter] * reg_b[counter];

          // store atb
          Atb(icol) = sum_atb;
        }

#ifdef HCAL_MAHI_GPUDEBUG
        printf("AtA\n");
        for (int i = 0; i < 8; i++) {
          for (int j = 0; j < 8; j++)
            printf("%f ", AtA(i, j));
          printf("\n");
        }
        printf("Atb\n");
        for (int i = 0; i < 8; i++)
          printf("%f ", Atb(i));
        printf("\n");
        printf("result Amplitudes before nnls\n");
        for (int i = 0; i < 8; i++)
          printf("%f ", resultAmplitudesVector(i));
        printf("\n");
#endif

        // for fnnls
        MapSymM<float, NPULSES> matrixLForFnnls{shrMatrixLFnnlsStorage};

        // run fast nnls
        // FIXME: provide values from config
        fnnls(AtA, Atb, resultAmplitudesVector, npassive, pulseOffsets, matrixLForFnnls, 1e-11, 500);

#ifdef HCAL_MAHI_GPUDEBUG
        printf("result Amplitudes\n");
        for (int i = 0; i < 8; i++)
          printf("resultAmplitudes(%d) = %f\n", i, resultAmplitudesVector(i));
#endif

        // replace pulseMatrixView * result - inputs
        // NOTE:
        float accum[NSAMPLES];
        Eigen::Map<ColumnVector<NSAMPLES>> mapAccum{accum};
        {
          float results[NPULSES];

// preload results and permute according to the pulse offsets
#pragma unroll
          for (int counter = 0; counter < NPULSES; counter++) {
            results[counter] = resultAmplitudesVector[counter];
          }

// load accum
#pragma unroll
          for (int counter = 0; counter < NSAMPLES; counter++)
            accum[counter] = -inputAmplitudesView(counter);

          // iterate
          for (int icol = 0; icol < NPULSES; icol++) {
            float pm_col[NSAMPLES];

// preload a column of pulse matrix
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              pm_col[counter] = __ldg(&glbPulseMatrixView.coeffRef(counter, icol));

// accum
#pragma unroll
            for (int counter = 0; counter < NSAMPLES; counter++)
              accum[counter] += results[icol] * pm_col[counter];
          }
        }

        // compute chi2 and check that there is no rotation
        //chi2 = matrixDecomposition
        //    .matrixL()
        //    . solve(mapAccum)
        //            .solve(pulseMatrixView * resultAmplitudesVector - inputAmplitudesView)
        //    .squaredNorm();
        {
          float reg_b_tmp[NSAMPLES];
          float reg_L[NSAMPLES];
          float accumSum = 0;

// preload a column and load column 0 of cholesky
#pragma unroll
          for (int i = 0; i < NSAMPLES; i++) {
            reg_b_tmp[i] = mapAccum(i);
            reg_L[i] = matrixL(i, 0);
          }

          // compute x0 and store it
          auto x_prev = reg_b_tmp[0] / reg_L[0];
          accumSum += x_prev * x_prev;

// iterate
#pragma unroll
          for (int iL = 1; iL < NSAMPLES; iL++) {
// update accum
#pragma unroll
            for (int counter = iL; counter < NSAMPLES; counter++)
              reg_b_tmp[counter] -= x_prev * reg_L[counter];

// load the next column of cholesky
#pragma unroll
            for (int counter = iL; counter < NSAMPLES; counter++)
              reg_L[counter] = matrixL(counter, iL);

            // compute the next x for M(iL, icol)
            x_prev = reg_b_tmp[iL] / reg_L[iL];

            // store the result value
            accumSum += x_prev * x_prev;
          }

          chi2 = accumSum;
        }

        auto const deltaChi2 = std::abs(chi2 - previous_chi2);
        if (chi2 == chi2_2itersback && chi2 < previous_chi2)
          break;

        // update
        chi2_2itersback = previous_chi2;
        previous_chi2 = chi2;

        // exit condition
        if (deltaChi2 < deltaChi2Threashold)
          break;
      }

#ifdef HCAL_MAHI_GPUDEBUG
      for (int i = 0; i < NPULSES; i++)
        printf("pulseOffsets(%d) = %d outputAmplitudes(%d) = %f\n", i, pulseOffsets(i), i, resultAmplitudesVector(i));
      printf("chi2 = %f\n", chi2);
#endif

      outputChi2[gch] = chi2;
      auto const idx_for_energy = std::abs(pulseOffsetValues[0]);
      outputEnergy[gch] = (gain * resultAmplitudesVector(idx_for_energy)) * respCorrection;
      /*
      #pragma unroll
      for (int i=0; i<NPULSES; i++)
          if (pulseOffsets[i] == soi)
              // NOTE: gain is a number < 10^-3/4, multiply first to avoid stab issues
              outputEnergy[gch] = (gain*resultAmplitudesVector(i))*respCorrection;
      */
    }

    void entryPoint(InputDataGPU const& inputGPU,
                    OutputDataGPU& outputGPU,
                    ConditionsProducts const& conditions,
                    ScratchDataGPU& scratch,
                    ConfigParameters const& configParameters,
                    cudaStream_t cudaStream) {
      auto const totalChannels = inputGPU.f01HEDigis.size + inputGPU.f5HBDigis.size + inputGPU.f3HBDigis.size;

      // FIXME: may be move this assignment to emphasize this more clearly
      // FIXME: number of channels for output might change given that
      //   some channesl might be filtered out
      outputGPU.recHits.size = totalChannels;

      // TODO: this can be lifted by implementing a separate kernel
      // similar to the default one, but properly handling the diff in #sample
      // or modifying existing one
      auto const f01nsamples = compute_nsamples<Flavor01>(inputGPU.f01HEDigis.stride);
      auto const f5nsamples = compute_nsamples<Flavor5>(inputGPU.f5HBDigis.stride);
      auto const f3nsamples = compute_nsamples<Flavor3>(inputGPU.f3HBDigis.stride);
      int constexpr windowSize = 8;
      int const startingSample = f01nsamples - windowSize;
      assert(startingSample==0 || startingSample==2);
      if (inputGPU.f01HEDigis.stride > 0 && inputGPU.f5HBDigis.stride> 0)
          assert(f01nsamples == f5nsamples);
      if (inputGPU.f01HEDigis.stride > 0 && inputGPU.f3HBDigis.stride > 0)
          assert(f01nsamples == f3nsamples);

      dim3 threadsPerBlock{windowSize, configParameters.kprep1dChannelsPerBlock};
      int blocks = static_cast<uint32_t>(threadsPerBlock.y) > totalChannels
                       ? 1
                       : (totalChannels + threadsPerBlock.y - 1) / threadsPerBlock.y;
      int nbytesShared =
          ((2 * windowSize + 2) * sizeof(float) + sizeof(uint64_t)) * configParameters.kprep1dChannelsPerBlock;
      kernel_prep1d_sameNumberOfSamples<<<blocks, threadsPerBlock, nbytesShared, cudaStream>>>(
          scratch.amplitudes.get(),
          scratch.noiseTerms.get(),
          outputGPU.recHits.energy.get(),
          outputGPU.recHits.chi2.get(),
          inputGPU.f01HEDigis.data.get(),
          inputGPU.f5HBDigis.data.get(),
          inputGPU.f3HBDigis.data.get(),
          inputGPU.f01HEDigis.ids.get(),
          inputGPU.f5HBDigis.ids.get(),
          inputGPU.f3HBDigis.ids.get(),
          inputGPU.f01HEDigis.stride,
          inputGPU.f5HBDigis.stride,
          inputGPU.f3HBDigis.stride,
          inputGPU.f01HEDigis.size,
          inputGPU.f5HBDigis.size,
          inputGPU.f5HBDigis.npresamples.get(),
          scratch.soiSamples.get(),
          outputGPU.recHits.energyM0.get(),
          outputGPU.recHits.timeM0.get(),
          outputGPU.recHits.did.get(),
          totalChannels,
          conditions.recoParams.param1,
          conditions.recoParams.param2,
          conditions.qieCoders.offsets,
          conditions.qieCoders.slopes,
          conditions.qieTypes.values,
          conditions.pedestalWidths.values,
          conditions.effectivePedestalWidths.values,
          conditions.pedestals.values,
          conditions.convertedEffectivePedestals ? conditions.convertedEffectivePedestals->values
                                                 : conditions.pedestals.values,
          configParameters.useEffectivePedestals,
          conditions.sipmParameters.type,
          conditions.sipmParameters.fcByPE,
          conditions.sipmCharacteristics.parLin1,
          conditions.sipmCharacteristics.parLin2,
          conditions.sipmCharacteristics.parLin3,
          conditions.gains.values,
          conditions.respCorrs.values,
          conditions.topology->maxDepthHB(),
          conditions.topology->maxDepthHE(),
          conditions.recConstants->getNPhi(1) > IPHI_MAX ? conditions.recConstants->getNPhi(1) : IPHI_MAX,
          conditions.topology->firstHBRing(),
          conditions.topology->lastHBRing(),
          conditions.topology->firstHERing(),
          conditions.topology->lastHERing(),
          conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
          conditions.topology->firstHERing() > conditions.topology->lastHERing()
              ? 0
              : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1),
          configParameters.sipmQTSShift,
          configParameters.sipmQNTStoSum,
          configParameters.firstSampleShift,
          conditions.offsetForHashes,
          configParameters.ts4Thresh,
          startingSample);
      cudaCheck(cudaGetLastError());

      // 1024 is the max threads per block for gtx1080
      // FIXME: take this from cuda service or something like that
      uint32_t const channelsPerBlock = 1024 / (windowSize * conditions.pulseOffsetsHost.size());
      dim3 threadsPerBlock2{windowSize, static_cast<uint32_t>(conditions.pulseOffsetsHost.size()), channelsPerBlock};
      int blocks2 =
          threadsPerBlock2.z > totalChannels ? 1 : (totalChannels + threadsPerBlock2.z - 1) / threadsPerBlock2.z;

#ifdef HCAL_MAHI_CPUDEBUG
      std::cout << "threads: " << threadsPerBlock2.x << " " << threadsPerBlock2.y << "  " << threadsPerBlock2.z
                << std::endl;
      std::cout << "blocks: " << blocks2 << std::endl;
#endif

      kernel_prep_pulseMatrices_sameNumberOfSamples<<<blocks2, threadsPerBlock2, 0, cudaStream>>>(
          scratch.pulseMatrices.get(),
          scratch.pulseMatricesM.get(),
          scratch.pulseMatricesP.get(),
          conditions.pulseOffsets.values,
          scratch.amplitudes.get(),
          inputGPU.f01HEDigis.ids.get(),
          inputGPU.f5HBDigis.ids.get(),
          inputGPU.f3HBDigis.ids.get(),
          inputGPU.f01HEDigis.size,
          inputGPU.f5HBDigis.size,
          totalChannels,
          scratch.soiSamples.get(),
          conditions.recoParams.ids,
          conditions.recoParams.acc25nsVec,
          conditions.recoParams.diff25nsItvlVec,
          conditions.recoParams.accVarLenIdxMinusOneVec,
          conditions.recoParams.diffVarItvlIdxMinusOneVec,
          conditions.recoParams.accVarLenIdxZEROVec,
          conditions.recoParams.diffVarItvlIdxZEROVec,
          configParameters.meanTime,
          configParameters.timeSigmaSiPM,
          configParameters.timeSigmaHPD,
          conditions.topology->maxDepthHB(),
          conditions.topology->maxDepthHE(),
          conditions.recConstants->getNPhi(1) > IPHI_MAX ? conditions.recConstants->getNPhi(1) : IPHI_MAX,
          conditions.topology->firstHBRing(),
          conditions.topology->lastHBRing(),
          conditions.topology->firstHERing(),
          conditions.topology->lastHERing(),
          conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
          conditions.topology->firstHERing() > conditions.topology->lastHERing()
              ? 0
              : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1),
          conditions.offsetForHashes,
          configParameters.applyTimeSlew,
          configParameters.tzeroTimeSlew,
          configParameters.slopeTimeSlew,
          configParameters.tmaxTimeSlew);
      cudaCheck(cudaGetLastError());

      // number of samples is checked in above assert
      if (conditions.pulseOffsetsHost.size() == 8u) {
        // FIXME: provide constants from configuration
        uint32_t threadsPerBlock = configParameters.kernelMinimizeThreads[0];
        uint32_t blocks = threadsPerBlock > totalChannels ? 1 : (totalChannels + threadsPerBlock - 1) / threadsPerBlock;
        auto const nbytesShared = 2 * threadsPerBlock * MapSymM<float, 8>::total * sizeof(float);
        kernel_minimize<8, 8><<<blocks, threadsPerBlock, nbytesShared, cudaStream>>>(
            outputGPU.recHits.energy.get(),
            outputGPU.recHits.chi2.get(),
            scratch.amplitudes.get(),
            scratch.pulseMatrices.get(),
            scratch.pulseMatricesM.get(),
            scratch.pulseMatricesP.get(),
            conditions.pulseOffsets.values,
            scratch.noiseTerms.get(),
            scratch.soiSamples.get(),
            conditions.pedestalWidths.values,
            conditions.effectivePedestalWidths.values,
            configParameters.useEffectivePedestals,
            inputGPU.f01HEDigis.ids.get(),
            inputGPU.f5HBDigis.ids.get(),
            inputGPU.f3HBDigis.ids.get(),
            conditions.gains.values,
            conditions.respCorrs.values,
            inputGPU.f01HEDigis.size,
            inputGPU.f5HBDigis.size,
            totalChannels,
            conditions.offsetForHashes,
            conditions.topology->maxDepthHB(),
            conditions.topology->maxDepthHE(),
            conditions.recConstants->getNPhi(1) > IPHI_MAX ? conditions.recConstants->getNPhi(1) : IPHI_MAX,
            conditions.topology->firstHBRing(),
            conditions.topology->lastHBRing(),
            conditions.topology->firstHERing(),
            conditions.topology->lastHERing(),
            conditions.recConstants->getEtaRange(0).second - conditions.recConstants->getEtaRange(0).first + 1,
            conditions.topology->firstHERing() > conditions.topology->lastHERing()
                ? 0
                : (conditions.topology->lastHERing() - conditions.topology->firstHERing() + 1));
      } else {
        throw cms::Exception("Invalid MahiGPU configuration")
            << "Currently support only 8 pulses and 8 time samples and provided: " << f01nsamples << " samples and "
            << conditions.pulseOffsetsHost.size() << " pulses" << std::endl;
      }
    }

  }  // namespace mahi
}  // namespace hcal
