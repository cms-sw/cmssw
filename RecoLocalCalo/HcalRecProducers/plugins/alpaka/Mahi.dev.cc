#include <iostream>
#include <alpaka/alpaka.hpp>
#include <Eigen/Dense>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/atomicMaxPair.h"

#include "DataFormats/CaloRecHit/interface/MultifitComputations.h"
// needed to compile with USER_CXXFLAGS="-DCOMPUTE_TDC_TIME"
#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
// TODO reuse some of the HCAL constats from
//#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConstants.h"

#include "Mahi.h"

#ifdef HCAL_MAHI_GPUDEBUG
#define DETID_TO_DEBUG 1125647428
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace hcal::reconstruction {
    namespace mahi {

      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float uint_as_float(uint32_t val) {
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
        return __uint_as_float(val);
#else
        return edm::bit_cast<float>(val);
#endif
      }

      ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t float_as_uint(float val) {
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
        return __float_as_uint(val);
#else
        return edm::bit_cast<unsigned int>(val);
#endif
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE float compute_time_slew_delay(float const fC,
                                                                   float const tzero,
                                                                   float const slope,
                                                                   float const tmax) {
        auto const rawDelay = tzero + slope * std::log(fC);
        return rawDelay < 0 ? 0 : (rawDelay > tmax ? tmax : rawDelay);
      }

      // HcalQIEShapes are hardcoded in HcalQIEData.cc basically
      // + some logic to generate 128 and 256 value arrays...
      ALPAKA_STATIC_ACC_MEM_CONSTANT float const qie8shape[129] = {
          -1,   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   16,
          18,   20,   22,   24,   26,   28,   31,   34,   37,   40,   44,   48,   52,   57,   62,   57,   62,
          67,   72,   77,   82,   87,   92,   97,   102,  107,  112,  117,  122,  127,  132,  142,  152,  162,
          172,  182,  192,  202,  217,  232,  247,  262,  282,  302,  322,  347,  372,  347,  372,  397,  422,
          447,  472,  497,  522,  547,  572,  597,  622,  647,  672,  697,  722,  772,  822,  872,  922,  972,
          1022, 1072, 1147, 1222, 1297, 1372, 1472, 1572, 1672, 1797, 1922, 1797, 1922, 2047, 2172, 2297, 2422,
          2547, 2672, 2797, 2922, 3047, 3172, 3297, 3422, 3547, 3672, 3922, 4172, 4422, 4672, 4922, 5172, 5422,
          5797, 6172, 6547, 6922, 7422, 7922, 8422, 9047, 9672, 10297};

      ALPAKA_STATIC_ACC_MEM_CONSTANT float const qie11shape[257] = {
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

      constexpr int32_t IPHI_MAX = 72;

      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t did2linearIndexHB(
          uint32_t const didraw, int const maxDepthHB, int const firstHBRing, int const lastHBRing, int const nEtaHB) {
        HcalDetId did{didraw};
        uint32_t const value = (did.depth() - 1) + maxDepthHB * (did.iphi() - 1);
        return did.ieta() > 0
                   ? value + maxDepthHB * hcal::reconstruction::mahi::IPHI_MAX * (did.ieta() - firstHBRing)
                   : value + maxDepthHB * hcal::reconstruction::mahi::IPHI_MAX * (did.ieta() + lastHBRing + nEtaHB);
      }
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t did2linearIndexHE(uint32_t const didraw,
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
      ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t get_qiecoder_index(uint32_t const capid, uint32_t const range) {
        return capid * 4 + range;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE float compute_reco_correction_factor(float const par1,
                                                                          float const par2,
                                                                          float const par3,
                                                                          float const x) {
        return par3 * x * x + par2 * x + par1;
      }

      // compute the charge using the adc, qie type and the appropriate qie shape array
      ALPAKA_FN_ACC ALPAKA_FN_INLINE float compute_coder_charge(
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

      ALPAKA_FN_ACC ALPAKA_FN_INLINE float compute_diff_charge_gain(int const qieType,
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

      using PulseShapeConstElement = typename HcalPulseShapeSoA::ConstView::const_element;
      // TODO: remove what's not needed
      // originally from from RecoLocalCalo/HcalRecAlgos/src/PulseShapeFunctor.cc
      ALPAKA_FN_ACC ALPAKA_FN_INLINE float compute_pulse_shape_value(PulseShapeConstElement const& pulseShape,
                                                                     float const pulse_time,
                                                                     int const sample,
                                                                     int const shift) {
        auto const& acc25nsVec = pulseShape.acc25nsVec();
        auto const& diff25nsItvlVec = pulseShape.diff25nsItvlVec();
        auto const& accVarLenIdxMinusOneVec = pulseShape.accVarLenIdxMinusOneVec();
        auto const& diffVarItvlIdxMinusOneVec = pulseShape.diffVarItvlIdxMinusOneVec();
        auto const& accVarLenIdxZeroVec = pulseShape.accVarLenIdxZEROVec();
        auto const& diffVarItvlIdxZeroVec = pulseShape.diffVarItvlIdxZEROVec();

        // constants
        constexpr float slew = 0.f;
        constexpr auto ns_per_bx = ::hcal::constants::nsPerBX;

        // FIXME: clean up all the rounding... this is coming from original cpu version
        float const i_start_float = -::hcal::constants::iniTimeShift - pulse_time - slew > 0.f
                                        ? 0.f
                                        : std::abs(-::hcal::constants::iniTimeShift - pulse_time - slew) + 1.f;
        int i_start = static_cast<int>(i_start_float);
        float offset_start = static_cast<float>(i_start) - ::hcal::constants::iniTimeShift - pulse_time - slew;

        // boundary
        if (offset_start == 1.0f) {
          offset_start = 0.f;
          i_start -= 1;
        }

        int const bin_start = static_cast<int>(offset_start);
        float const bin_start_up = static_cast<float>(bin_start) + 0.5f;
        int const bin_0_start = offset_start < bin_start_up ? bin_start - 1 : bin_start;
        int const its_start = i_start / ns_per_bx;
        int const distTo25ns_start = ns_per_bx - 1 - i_start % ns_per_bx;
        auto const factor = offset_start - static_cast<float>(bin_0_start) - 0.5;

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

      // TODO: provide constants from configuration
      // from RecoLocalCalo/HcalRecProducers/python/HBHEMahiParameters_cfi.py
      constexpr int nMaxItersMin = 50;
      constexpr int nMaxItersNNLS = 500;
      constexpr double nnlsThresh = 1e-11;
      constexpr float deltaChi2Threashold = 1e-3;

      // from RecoLocalCalo/HcalRecProducers/src/HBHEPhase1Reconstructor.cc
      ALPAKA_FN_ACC float get_raw_charge(double const charge,
                                         double const pedestal,
                                         float const* shrChargeMinusPedestal,
                                         float const parLin1,
                                         float const parLin2,
                                         float const parLin3,
                                         int32_t const nsamplesForCompute,
                                         int32_t const soi,
                                         int const sipmQTSShift,
                                         int const sipmQNTStoSum,
                                         float const fcByPE,
                                         int32_t const lch,
                                         bool const isqie11) {
        float rawCharge;

        if (!isqie11)
          rawCharge = charge;
        else {
          int const first = std::max(soi + sipmQTSShift, 0);
          int const last = std::min(soi + sipmQNTStoSum, nsamplesForCompute);
          float sipmq = 0.0f;
          for (auto ts = first; ts < last; ts++)
            sipmq += shrChargeMinusPedestal[lch * nsamplesForCompute + ts];
          auto const effectivePixelsFired = sipmq / fcByPE;
          auto const factor = compute_reco_correction_factor(parLin1, parLin2, parLin3, effectivePixelsFired);
          rawCharge = (charge - pedestal) * factor + pedestal;

#ifdef HCAL_MAHI_GPUDEBUG
          printf("first = %d last = %d sipmQ = %f factor = %f rawCharge = %f\n", first, last, sipmq, factor, rawCharge);
#endif
        }
        return rawCharge;
      }

      // Tried using lambda, but strange error from  with nvcc
      inline constexpr bool TSenergyCompare(std::pair<unsigned int, float> a, std::pair<unsigned int, float> b) {
        return a.second > b.second;
      };

      class Kernel_prep1d_sameNumberOfSamples {
      public:
        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                      OProductType::View outputGPU,
                                      IProductTypef01::ConstView f01HEDigis,
                                      IProductTypef5::ConstView f5HBDigis,
                                      IProductTypef3::ConstView f3HBDigis,
                                      HcalMahiConditionsPortableDevice::ConstView mahi,
                                      HcalSiPMCharacteristicsPortableDevice::ConstView sipmCharacteristics,
                                      HcalRecoParamWithPulseShapeDevice::ConstView recoParamsWithPS,
                                      bool const useEffectivePedestals,
                                      int const sipmQTSShift,
                                      int const sipmQNTStoSum,
                                      int const firstSampleShift,
                                      float const ts4Thresh,
                                      float* amplitudes,
                                      float* noiseTerms,
                                      float* electronicNoiseTerms,
                                      int8_t* soiSamples,
                                      int const windowSize) const {
          auto const nchannelsf015 = f01HEDigis.size() + f5HBDigis.size();
          auto const startingSample = compute_nsamples<Flavor1>(f01HEDigis.stride()) - windowSize;
          auto const nchannels = f01HEDigis.size() + f5HBDigis.size() + f3HBDigis.size();

          //first index = groups of channel
          auto const nchannels_per_block(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
          //2nd index = groups of sample
          auto const nsamplesForCompute(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[1u]);

          // configure shared mem
          float* shrEnergyM0PerTS = alpaka::getDynSharedMem<float>(acc);
          float* shrChargeMinusPedestal = shrEnergyM0PerTS + nsamplesForCompute * nchannels_per_block;
          float* shrMethod0EnergyAccum = shrChargeMinusPedestal + nsamplesForCompute * nchannels_per_block;
          float* shrEnergyM0TotalAccum = shrMethod0EnergyAccum + nchannels_per_block;
          unsigned long long int* shrMethod0EnergySamplePair =
              reinterpret_cast<unsigned long long int*>(shrEnergyM0TotalAccum + nchannels_per_block);

          //Loop over all groups of channels
          for (auto group : uniform_groups_y(acc, nchannels)) {
            //Loop over each channel, first compute soiSamples and shrMem for all channels
            for (auto channel : uniform_group_elements_y(acc, group, nchannels)) {
              auto const gch = channel.global;
              auto const lch = channel.local;

              for (auto i_sample : independent_group_elements_x(acc, nsamplesForCompute)) {
                auto const sampleWithinWindow = i_sample;
                auto const sample = i_sample + startingSample;
                auto const linearThPerBlock = i_sample + channel.local * nsamplesForCompute;
                // initialize
                if (sampleWithinWindow == 0) {
                  soiSamples[gch] = -1;
                  shrMethod0EnergyAccum[lch] = 0;
                  shrMethod0EnergySamplePair[lch] = 0;
                  shrEnergyM0TotalAccum[lch] = 0;
                }

                // compute soiSamples
                if (gch < f01HEDigis.size()) {
                  // NOTE: assume that soi is high only for a single guy!
                  //   which must be the case. cpu version does not check for that
                  //   if that is not the case, we will see that with cuda mmecheck
                  auto const soibit = soibit_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample);
                  if (soibit == 1)
                    soiSamples[gch] = sampleWithinWindow;
                } else if (gch >= nchannelsf015) {
                  auto const soibit = soibit_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample);
                  if (soibit == 1)
                    soiSamples[gch] = sampleWithinWindow;
                }

                // compute shrMem
                auto const id = gch < f01HEDigis.size()
                                    ? f01HEDigis.ids()[gch]
                                    : (gch < nchannelsf015 ? f5HBDigis.ids()[gch - f01HEDigis.size()]
                                                           : f3HBDigis.ids()[gch - nchannelsf015]);
                auto const did = HcalDetId{id};

                auto const adc =
                    gch < f01HEDigis.size()
                        ? adc_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample)
                        : (gch < nchannelsf015
                               ? adc_for_sample<Flavor5>(&(f5HBDigis.data()[gch - f01HEDigis.size()][0]), sample)
                               : adc_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample));
                auto const capid =
                    gch < f01HEDigis.size()
                        ? capid_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample)
                        : (gch < nchannelsf015
                               ? capid_for_sample<Flavor5>(&(f5HBDigis.data()[gch - f01HEDigis.size()][0]), sample)
                               : capid_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample));

                // compute hash for this did
                // hash needed to convert condition arrays order (based on Topo) into digi arrays order(based on FED)
                auto const hashedId =
                    did.subdetId() == HcalBarrel
                        ? did2linearIndexHB(id, mahi.maxDepthHB(), mahi.firstHBRing(), mahi.lastHBRing(), mahi.nEtaHB())
                        : did2linearIndexHE(id,
                                            mahi.maxDepthHE(),
                                            mahi.maxPhiHE(),
                                            mahi.firstHERing(),
                                            mahi.lastHERing(),
                                            mahi.nEtaHE()) +
                              mahi.offsetForHashes();

                // conditions based on the hash
                auto const qieType = mahi.qieTypes_values()[hashedId] > 0 ? 1 : 0;  // 2 types at this point
                auto const* qieOffsets = mahi.qieCoders_offsets()[hashedId].data();
                auto const* qieSlopes = mahi.qieCoders_slopes()[hashedId].data();
                auto const pedestal = mahi.pedestals_value()[hashedId][capid];

                // compute charge
                auto const charge = compute_coder_charge(qieType, adc, capid, qieOffsets, qieSlopes);

                shrChargeMinusPedestal[linearThPerBlock] = charge - pedestal;
              }
            }
            alpaka::syncBlockThreads(acc);

            //Loop over each channel, compute input for multifit using shrChargeMinusPedestal
            for (auto channel : uniform_group_elements_y(acc, group, nchannels)) {
              auto const gch = channel.global;
              auto const lch = channel.local;
              for (auto i_sample : independent_group_elements_x(acc, nsamplesForCompute)) {
                auto const sampleWithinWindow = i_sample;
                auto const sample = i_sample + startingSample;

                // initialize all output buffers
                if (sampleWithinWindow == 0) {
                  outputGPU.detId()[gch] = 0;
                  outputGPU.energyM0()[gch] = 0;
                  outputGPU.timeM0()[gch] = 0;
                  outputGPU.energy()[gch] = 0;
                  outputGPU.chi2()[gch] = 0;
                }

                // offset output
                auto* amplitudesForChannel = amplitudes + nsamplesForCompute * gch;
                auto* noiseTermsForChannel = noiseTerms + nsamplesForCompute * gch;
                auto* electronicNoiseTermsForChannel = electronicNoiseTerms + nsamplesForCompute * gch;

                // get event input quantities
                auto const stride = gch < f01HEDigis.size()
                                        ? f01HEDigis.stride()
                                        : (gch < f5HBDigis.size() ? f5HBDigis.stride() : f3HBDigis.stride());
                auto const nsamples =
                    gch < f01HEDigis.size()
                        ? compute_nsamples<Flavor1>(stride)
                        : (gch < nchannelsf015 ? compute_nsamples<Flavor5>(stride) : compute_nsamples<Flavor3>(stride));

                ALPAKA_ASSERT_ACC(nsamples == nsamplesForCompute || nsamples - startingSample == nsamplesForCompute);

                auto const id = gch < f01HEDigis.size()
                                    ? f01HEDigis.ids()[gch]
                                    : (gch < nchannelsf015 ? f5HBDigis.ids()[gch - f01HEDigis.size()]
                                                           : f3HBDigis.ids()[gch - nchannelsf015]);
                auto const did = HcalDetId{id};

                auto const adc =
                    gch < f01HEDigis.size()
                        ? adc_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample)
                        : (gch < nchannelsf015
                               ? adc_for_sample<Flavor5>(&(f5HBDigis.data()[gch - f01HEDigis.size()][0]), sample)
                               : adc_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample));
                auto const capid =
                    gch < f01HEDigis.size()
                        ? capid_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample)
                        : (gch < nchannelsf015
                               ? capid_for_sample<Flavor5>(&(f5HBDigis.data()[gch - f01HEDigis.size()][0]), sample)
                               : capid_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample));

                // compute hash for this did
                // hash needed to convert condition arrays order (based on Topo) into digi arrays order(based on FED)
                auto const hashedId =
                    did.subdetId() == HcalBarrel
                        ? did2linearIndexHB(id, mahi.maxDepthHB(), mahi.firstHBRing(), mahi.lastHBRing(), mahi.nEtaHB())
                        : did2linearIndexHE(id,
                                            mahi.maxDepthHE(),
                                            mahi.maxPhiHE(),
                                            mahi.firstHERing(),
                                            mahi.lastHERing(),
                                            mahi.nEtaHE()) +
                              mahi.offsetForHashes();

                // conditions based on the hash
                auto const qieType = mahi.qieTypes_values()[hashedId] > 0 ? 1 : 0;  // 2 types at this point
                auto const* qieOffsets = mahi.qieCoders_offsets()[hashedId].data();
                auto const* qieSlopes = mahi.qieCoders_slopes()[hashedId].data();
                auto const* pedestalWidthsForChannel =
                    useEffectivePedestals && (gch < f01HEDigis.size() || gch >= nchannelsf015)
                        ? mahi.effectivePedestalWidths()[hashedId].data()
                        : mahi.pedestals_width()[hashedId].data();

                auto const gain = mahi.gains_value()[hashedId][capid];
                auto const gain0 = mahi.gains_value()[hashedId][0];
                auto const respCorrection = mahi.respCorrs_values()[hashedId];
                auto const pedestal = mahi.pedestals_value()[hashedId][capid];
                auto const pedestalWidth = pedestalWidthsForChannel[capid];
                // if needed, only use effective pedestals for f01
                auto const pedestalToUseForMethod0 =
                    useEffectivePedestals && (gch < f01HEDigis.size() || gch >= nchannelsf015)
                        ? mahi.effectivePedestals()[hashedId][capid]
                        : pedestal;
                auto const sipmType = mahi.sipmPar_type()[hashedId];
                auto const fcByPE = mahi.sipmPar_fcByPE()[hashedId];
                auto const recoParam1 = recoParamsWithPS.recoParamView().param1()[hashedId];
                auto const recoParam2 = recoParamsWithPS.recoParamView().param2()[hashedId];

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

                int32_t const soi =
                    gch < f01HEDigis.size()
                        ? soiSamples[gch]
                        : (gch < nchannelsf015 ? f5HBDigis.npresamples()[gch - f01HEDigis.size()] : soiSamples[gch]);

                bool badSOI = (soi < 0 or static_cast<unsigned>(soi) >= nsamplesForCompute);
                if (badSOI and sampleWithinWindow == 0) {
#ifdef GPU_DEBUG
                  printf("Found HBHE channel %d with invalid SOI %d\n", gch, soi);
#endif
                  // mark the channel as bad
                  outputGPU.chi2()[gch] = -9999.f;
                }

                // type index starts from 1 .. 6
                // precicionItem index starts from 0 .. 5
                auto const precisionItem = sipmCharacteristics.precisionItem()[sipmType - 1];
                auto const parLin1 = precisionItem.parLin1_;
                auto const parLin2 = precisionItem.parLin2_;
                auto const parLin3 = precisionItem.parLin3_;

                //int32_t const soi = gch >= nchannelsf01HE
                //    ? npresamplesf5HB[gch - nchannelsf01HE]
                //    : soiSamples[gch];
                // this is here just to make things uniform...
                if (gch >= f01HEDigis.size() && gch < nchannelsf015 && sampleWithinWindow == 0)
                  soiSamples[gch] = f5HBDigis.npresamples()[gch - f01HEDigis.size()];

                //
                // compute various quantities (raw charge and tdc stuff)
                // NOTE: this branch will be divergent only for a single warp that
                // sits on the boundary when flavor 01 channels end and flavor 5 start
                //
                float const rawCharge = get_raw_charge(charge,
                                                       pedestal,
                                                       shrChargeMinusPedestal,
                                                       parLin1,
                                                       parLin2,
                                                       parLin3,
                                                       nsamplesForCompute,
                                                       soi,
                                                       sipmQTSShift,
                                                       sipmQNTStoSum,
                                                       fcByPE,
                                                       lch,
                                                       gch < f01HEDigis.size() || gch >= nchannelsf015);

                auto const dfc = compute_diff_charge_gain(
                    qieType, adc, capid, qieOffsets, qieSlopes, gch < f01HEDigis.size() || gch >= nchannelsf015);

#ifdef COMPUTE_TDC_TIME
                float tdcTime;
                if (gch >= f01HEDigis.size() && gch < nchannelsf015) {
                  tdcTime = HcalSpecialTimes::UNKNOWN_T_NOTDC;
                } else {
                  if (gch < f01HEDigis.size())
                    tdcTime =
                        HcalSpecialTimes::getTDCTime(tdc_for_sample<Flavor1>(&(f01HEDigis.data()[gch][0]), sample));
                  else if (gch >= nchannelsf015)
                    tdcTime = HcalSpecialTimes::getTDCTime(
                        tdc_for_sample<Flavor3>(&(f3HBDigis.data()[gch - nchannelsf015][0]), sample));
                }
#endif  // COMPUTE_TDC_TIME

                // compute method 0 quantities
                // TODO: need to apply containment
                // TODO: need to apply time slew
                // TODO: for < run 3, apply HBM legacy energy correction
                auto const nsamplesToAdd = recoParam1 < 10 ? recoParam2 : (recoParam1 >> 14) & 0xF;
                auto const startSampleTmp = soi + firstSampleShift;
                auto const startSample = startSampleTmp < 0 ? 0 : startSampleTmp;
                auto const endSample =
                    startSample + nsamplesToAdd < nsamplesForCompute ? startSample + nsamplesToAdd : nsamplesForCompute;
                // NOTE: gain is a small number < 10^-3, multiply it last
                auto const energym0_per_ts = gain * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
                auto const energym0_per_ts_gain0 = gain0 * ((rawCharge - pedestalToUseForMethod0) * respCorrection);
                // store to shared mem
                shrEnergyM0PerTS[lch * nsamplesForCompute + sampleWithinWindow] = energym0_per_ts;

                alpaka::atomicAdd(
                    acc, &shrEnergyM0TotalAccum[lch], energym0_per_ts_gain0, alpaka::hierarchy::Threads{});

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
                printf("startSample = %d endSample = %d param1 = %u param2 = %u\n",
                       startSample,
                       endSample,
                       recoParam1,
                       recoParam2);
#endif
                //Find the max energy of lch channel and the corresponding TS
                if (sampleWithinWindow >= static_cast<unsigned>(startSample) &&
                    sampleWithinWindow < static_cast<unsigned>(endSample)) {
                  //sum the energys of all TS
                  alpaka::atomicAdd(acc, &shrMethod0EnergyAccum[lch], energym0_per_ts, alpaka::hierarchy::Threads{});
                  // pair TS and E for all TSs, find max pair according to E
                  // TODO: Non-deterministic behavior for TS with equal energy
                  atomicMaxPair(
                      acc, &shrMethod0EnergySamplePair[lch], {sampleWithinWindow, energym0_per_ts}, TSenergyCompare);
                }

                // NOTE: must take soi, as values for that thread are used...
                // NOTE: does not run if soi is bad, because it does not match any sampleWithinWindow
                if (sampleWithinWindow == static_cast<unsigned>(soi)) {
                  // Channel quality check
                  //    https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HcalRecAlgos/plugins/HcalChannelPropertiesEP.cc#L107-L109
                  //    https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/CondCore/HcalPlugins/plugins/HcalChannelQuality_PayloadInspector.cc#L30
                  //      const bool taggedBadByDb = severity.dropChannel(digistatus->getValue());
                  //  do not run MAHI if taggedBadByDb = true
                  auto const digiStatus_ = mahi.channelQuality_status()[hashedId];
                  const bool taggedBadByDb = (digiStatus_ / 32770);

                  if (taggedBadByDb)
                    outputGPU.chi2()[gch] = -9999.f;

                  // check as in cpu version if mahi is not needed
                  // (use "not" and ">", instead of "<=", to ensure that a NaN value will pass the check, and the hit be flagged as invalid)
                  if (not(energym0_per_ts_gain0 > ts4Thresh)) {
                    outputGPU.chi2()[gch] = -9999.f;
                  }
                }
                //
                // preparations for mahi fit
                //
                auto const amplitude = rawCharge - pedestalToUseForMethod0;
                auto const noiseADC = (1. / std::sqrt(12)) * dfc;
                auto const noisePhotoSq = amplitude > pedestalWidth ? (amplitude * fcByPE) : 0.f;
                auto const noiseTerm = noiseADC * noiseADC + noisePhotoSq + pedestalWidth * pedestalWidth;

                // store to global memory
                amplitudesForChannel[sampleWithinWindow] = amplitude;
                noiseTermsForChannel[sampleWithinWindow] = noiseTerm;
                electronicNoiseTermsForChannel[sampleWithinWindow] = pedestalWidth;

              }  //end sample loop
            }  // end channel loop
            alpaka::syncBlockThreads(acc);

            // compute energy using shrMethod0EnergySamplePair
            for (auto channel : uniform_group_elements_y(acc, group, nchannels)) {
              auto const gch = channel.global;
              auto const lch = channel.local;
              for (auto i_sample : independent_group_elements_x(acc, nsamplesForCompute)) {
                auto const sampleWithinWindow = i_sample;

                int32_t const soi =
                    gch < f01HEDigis.size()
                        ? soiSamples[gch]
                        : (gch < nchannelsf015 ? f5HBDigis.npresamples()[gch - f01HEDigis.size()] : soiSamples[gch]);

                auto const id = gch < f01HEDigis.size()
                                    ? f01HEDigis.ids()[gch]
                                    : (gch < nchannelsf015 ? f5HBDigis.ids()[gch - f01HEDigis.size()]
                                                           : f3HBDigis.ids()[gch - nchannelsf015]);
                auto const did = HcalDetId{id};
                auto const hashedId =
                    did.subdetId() == HcalBarrel
                        ? did2linearIndexHB(id, mahi.maxDepthHB(), mahi.firstHBRing(), mahi.lastHBRing(), mahi.nEtaHB())
                        : did2linearIndexHE(id,
                                            mahi.maxDepthHE(),
                                            mahi.maxPhiHE(),
                                            mahi.firstHERing(),
                                            mahi.lastHERing(),
                                            mahi.nEtaHE()) +
                              mahi.offsetForHashes();

                auto const recoParam1 = recoParamsWithPS.recoParamView().param1()[hashedId];
                auto const recoParam2 = recoParamsWithPS.recoParamView().param2()[hashedId];

                auto const nsamplesToAdd = recoParam1 < 10 ? recoParam2 : (recoParam1 >> 14) & 0xF;
                // NOTE: must take soi, as values for that thread are used...
                // NOTE: does not run if soi is bad, because it does not match any sampleWithinWindow
                if (sampleWithinWindow == static_cast<unsigned>(soi)) {
                  auto const method0_energy = shrMethod0EnergyAccum[lch];
                  auto const val = shrMethod0EnergySamplePair[lch];
                  int const max_sample = (val >> 32) & 0xffffffff;

                  float const max_energy = uint_as_float(static_cast<uint32_t>(val & 0xffffffff));

                  float const max_energy_1 = static_cast<unsigned>(max_sample) < nsamplesForCompute - 1
                                                 ? shrEnergyM0PerTS[lch * nsamplesForCompute + max_sample + 1]
                                                 : 0.f;
                  float const position = nsamplesToAdd < nsamplesForCompute ? max_sample - soi : max_sample;
                  auto const sum = max_energy + max_energy_1;
                  // FIXME: for full comparison with cpu method 0  timing,
                  // need to correct by slew
                  // requires an accumulator -> more shared mem -> omit here unless
                  // really needed
                  float const time =
                      max_energy > 0.f && max_energy_1 > 0.f ? 25.f * (position + max_energy_1 / sum) : 25.f * position;

                  // store method0 quantities to global mem
                  outputGPU.detId()[gch] = id;
                  outputGPU.energyM0()[gch] = method0_energy;
                  outputGPU.timeM0()[gch] = time;

                  // check as in cpu version if mahi is not needed
                  // FIXME: KNOWN ISSUE: observed a problem when rawCharge and pedestal
                  // are basically equal and generate -0.00000...
                  // needs to be treated properly
                  // (use "not" and ">", instead of "<=", to ensure that a NaN value will pass the check, and the hit be flagged as invalid)
                  if (not(shrEnergyM0TotalAccum[lch] > 0)) {
                    outputGPU.chi2()[gch] = -9999.f;
                  }

#ifdef HCAL_MAHI_GPUDEBUG
                  printf("tsTOT = %f tstrig = %f ts4Thresh = %f\n",
                         shrEnergyM0TotalAccum[lch],
                         energym0_per_ts_gain0,
                         ts4Thresh);
#endif

#ifdef HCAL_MAHI_GPUDEBUG
                  printf(" method0_energy = %f max_sample = %d max_energy = %f time = %f\n",
                         method0_energy,
                         max_sample,
                         max_energy,
                         time);
#endif
                }
#ifdef HCAL_MAHI_GPUDEBUG
                printf(
                    "charge(%d) = %f pedestal(%d) = %f dfc(%d) = %f pedestalWidth(%d) = %f noiseADC(%d) = %f "
                    "noisPhoto(%d) =%f outputGPU chi2[gch] = %f \n",
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
                    noisePhotoSq,
                    outputGPU.chi2()[gch]);
#endif

              }  // loop for sample
            }  // loop for channels
          }  // loop for channgel groups
        }
      };  //Kernel_prep1d_sameNumberOfSamples

      class Kernel_prep_pulseMatrices_sameNumberOfSamples {
      public:
        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                      float* pulseMatrices,
                                      float* pulseMatricesM,
                                      float* pulseMatricesP,
                                      HcalMahiPulseOffsetsSoA::ConstView pulseOffsets,
                                      float const* amplitudes,
                                      IProductTypef01::ConstView f01HEDigis,
                                      IProductTypef5::ConstView f5HBDigis,
                                      IProductTypef3::ConstView f3HBDigis,
                                      int8_t* soiSamples,
                                      HcalMahiConditionsPortableDevice::ConstView mahi,
                                      HcalRecoParamWithPulseShapeDevice::ConstView recoParamsWithPS,
                                      float const meanTime,
                                      float const timeSigmaSiPM,
                                      float const timeSigmaHPD,
                                      bool const applyTimeSlew,
                                      float const tzeroTimeSlew,
                                      float const slopeTimeSlew,
                                      float const tmaxTimeSlew) const {
          //2nd index = pulse
          auto const npulses(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[1u]);
          //3rd index = sample
          auto const nsamples(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[2u]);

          auto const nchannels = f01HEDigis.size() + f5HBDigis.size() + f3HBDigis.size();
          auto const nchannelsf015 = f01HEDigis.size() + f5HBDigis.size();

          //Loop over each channel
          for (auto channel : uniform_elements_z(acc, nchannels)) {
            //Loop over pulses
            for (auto ipulse : independent_group_elements_y(acc, npulses)) {
              //Loop over sample
              for (auto sample : independent_group_elements_x(acc, nsamples)) {
                // conditions
                auto const id = channel < f01HEDigis.size()
                                    ? f01HEDigis.ids()[channel]
                                    : (channel < nchannelsf015 ? f5HBDigis.ids()[channel - f01HEDigis.size()]
                                                               : f3HBDigis.ids()[channel - nchannelsf015]);
                auto const deltaT =
                    channel >= f01HEDigis.size() && channel < nchannelsf015 ? timeSigmaHPD : timeSigmaSiPM;

                // compute hash for this did
                // hash needed to convert condition arrays order (based on Topo) into digi arrays order(based on FED)
                auto const did = DetId{id};
                auto const hashedId =
                    did.subdetId() == HcalBarrel
                        ? did2linearIndexHB(id, mahi.maxDepthHB(), mahi.firstHBRing(), mahi.lastHBRing(), mahi.nEtaHB())
                        : did2linearIndexHE(id,
                                            mahi.maxDepthHE(),
                                            mahi.maxPhiHE(),
                                            mahi.firstHERing(),
                                            mahi.lastHERing(),
                                            mahi.nEtaHE()) +
                              mahi.offsetForHashes();
                auto const pulseShape = recoParamsWithPS.getPulseShape(hashedId);

                // offset output arrays
                auto* pulseMatrix = pulseMatrices + nsamples * npulses * channel;
                auto* pulseMatrixM = pulseMatricesM + nsamples * npulses * channel;
                auto* pulseMatrixP = pulseMatricesP + nsamples * npulses * channel;

                // amplitude per ipulse
                int const soi = soiSamples[channel];
                int const pulseOffset = pulseOffsets.offsets()[ipulse];
                auto const amplitude = amplitudes[channel * nsamples + pulseOffset + soi];

                if (amplitude <= 0.f) {
                  pulseMatrix[ipulse * nsamples + sample] = 0.f;
                  pulseMatrixM[ipulse * nsamples + sample] = 0.f;
                  pulseMatrixP[ipulse * nsamples + sample] = 0.f;
                  continue;
                }
#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
                if (id != DETID_TO_DEBUG)
                  return;
#endif
                if (sample == 0 && ipulse == 0) {
                  for (int i = 0; i < 8; i++)
                    printf("amplitude(%d) = %f\n", i, amplitudes[channel * nsamples + i]);
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
                  for (int i = 0; i < hcal::constants::maxSamples; i++) {
                    auto const value = compute_pulse_shape_value(pulseShape, t0, i, 0);
                  }
                  printf("\n");
                  for (int i = 0; i < hcal::constants::maxSamples; i++) {
                    auto const value = compute_pulse_shape_value(pulseShape, t0p, i, 0);
                  }
                  printf("\n");
                  for (int i = 0; i < hcal::constants::maxSamples; i++) {
                    auto const value = compute_pulse_shape_value(pulseShape, t0m, i, 0);
                  }
                }
#endif

                // FIXME: shift should be treated properly,
                // here assume 8 time slices and 8 samples
                auto const shift = 4 - soi;  // as in cpu version!

                int32_t const idx = sample - pulseOffset;
                auto const value = idx >= 0 && static_cast<unsigned>(idx) < nsamples
                                       ? compute_pulse_shape_value(pulseShape, t0, idx, shift)
                                       : 0;
                auto const value_t0m = idx >= 0 && static_cast<unsigned>(idx) < nsamples
                                           ? compute_pulse_shape_value(pulseShape, t0m, idx, shift)
                                           : 0;
                auto const value_t0p = idx >= 0 && static_cast<unsigned>(idx) < nsamples
                                           ? compute_pulse_shape_value(pulseShape, t0p, idx, shift)
                                           : 0;

                // store to global
                pulseMatrix[ipulse * nsamples + sample] = value;
                pulseMatrixM[ipulse * nsamples + sample] = value_t0m;
                pulseMatrixP[ipulse * nsamples + sample] = value_t0p;

              }  // loop over sample
            }  // loop over pulse
          }  // loop over channels
        }
      };  // Kernel_prep_pulseMatrices_sameNumberOfSamples

      template <int NSAMPLES, int NPULSES>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void update_covariance(
          calo::multifit::ColumnVector<NPULSES> const& resultAmplitudesVector,
          calo::multifit::MapSymM<float, NSAMPLES>& covarianceMatrix,
          Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrix,
          Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixM,
          Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> const& pulseMatrixP) {
        CMS_UNROLL_LOOP
        for (int ipulse = 0; ipulse < NPULSES; ipulse++) {
          auto const resultAmplitude = resultAmplitudesVector(ipulse);
          if (resultAmplitude == 0)
            continue;

#ifdef HCAL_MAHI_GPUDEBUG
          printf("pulse cov array for ibx = %d\n", ipulse);
#endif

          // preload a column
          float pmcol[NSAMPLES], pmpcol[NSAMPLES], pmmcol[NSAMPLES];
          CMS_UNROLL_LOOP
          for (int counter = 0; counter < NSAMPLES; counter++) {
            pmcol[counter] = pulseMatrix.coeffRef(counter, ipulse);
            pmpcol[counter] = pulseMatrixP.coeffRef(counter, ipulse);
            pmmcol[counter] = pulseMatrixM.coeffRef(counter, ipulse);
          }

          auto const ampl2 = resultAmplitude * resultAmplitude;
          CMS_UNROLL_LOOP
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
            CMS_UNROLL_LOOP
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
      class Kernel_minimize {
      public:
        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                      OProductType::View outputGPU,
                                      float const* amplitudes,
                                      float* pulseMatrices,
                                      float* pulseMatricesM,
                                      float* pulseMatricesP,
                                      HcalMahiPulseOffsetsSoA::ConstView pulseOffsetsView,
                                      float* noiseTerms,
                                      float* electronicNoiseTerms,
                                      int8_t* soiSamples,
                                      HcalMahiConditionsPortableDevice::ConstView mahi,
                                      bool const useEffectivePedestals,
                                      IProductTypef01::ConstView f01HEDigis,
                                      IProductTypef5::ConstView f5HBDigis,
                                      IProductTypef3::ConstView f3HBDigis) const {
          // can be relaxed if needed - minor updates are needed in that case!
          static_assert(NPULSES == NSAMPLES);

          auto const nchannels = f01HEDigis.size() + f5HBDigis.size() + f3HBDigis.size();

          auto const nchannelsf015 = f01HEDigis.size() + f5HBDigis.size();

          auto const threadsPerBlock(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

          //Loop over all groups of channels
          for (auto group : uniform_groups_x(acc, nchannels)) {
            //Loop over each channel
            for (auto channel : uniform_group_elements_x(acc, group, nchannels)) {
              auto const gch = channel.global;
              auto const lch = channel.local;

              // if chi2 is set to -9999 do not run minimization
              if (outputGPU.chi2()[gch] == -9999.f)
                continue;

              // configure shared mem
              float* shrmem = alpaka::getDynSharedMem<float>(acc);
              float* shrMatrixLFnnlsStorage = shrmem + calo::multifit::MapSymM<float, NPULSES>::total * lch;
              float* shrAtAStorage = shrmem + calo::multifit::MapSymM<float, NPULSES>::total * (lch + threadsPerBlock);

              // conditions for pedestal widths
              auto const id = gch < f01HEDigis.size() ? f01HEDigis.ids()[gch]
                                                      : (gch < nchannelsf015 ? f5HBDigis.ids()[gch - f01HEDigis.size()]
                                                                             : f3HBDigis.ids()[gch - nchannelsf015]);
              auto const did = DetId{id};
              auto const hashedId =
                  did.subdetId() == HcalBarrel
                      ? did2linearIndexHB(id, mahi.maxDepthHB(), mahi.firstHBRing(), mahi.lastHBRing(), mahi.nEtaHB())
                      : did2linearIndexHE(id,
                                          mahi.maxDepthHE(),
                                          mahi.maxPhiHE(),
                                          mahi.firstHERing(),
                                          mahi.lastHERing(),
                                          mahi.nEtaHE()) +
                            mahi.offsetForHashes();

              // conditions based on the hash
              auto const* pedestalWidthsForChannel =
                  useEffectivePedestals && (gch < f01HEDigis.size() || gch >= nchannelsf015)
                      ? mahi.effectivePedestalWidths()[hashedId].data()
                      : mahi.pedestals_width()[hashedId].data();
              auto const averagePedestalWidth2 = 0.25 * (pedestalWidthsForChannel[0] * pedestalWidthsForChannel[0] +
                                                         pedestalWidthsForChannel[1] * pedestalWidthsForChannel[1] +
                                                         pedestalWidthsForChannel[2] * pedestalWidthsForChannel[2] +
                                                         pedestalWidthsForChannel[3] * pedestalWidthsForChannel[3]);

              // FIXME on cpu ts 0 capid was used - does it make any difference
              auto const gain = mahi.gains_value()[hashedId][0];

              auto const respCorrection = mahi.respCorrs_values()[hashedId];
              auto const noisecorr = mahi.sipmPar_auxi2()[hashedId];

#ifdef HCAL_MAHI_GPUDEBUG
#ifdef HCAL_MAHI_GPUDEBUG_FILTERDETID
              if (id != DETID_TO_DEBUG)
                return;
#endif
#endif

              calo::multifit::ColumnVector<NPULSES, int> pulseOffsets;
              CMS_UNROLL_LOOP
              for (int i = 0; i < NPULSES; ++i)
                pulseOffsets(i) = i;

              // output amplitudes/weights
              calo::multifit::ColumnVector<NPULSES> resultAmplitudesVector =
                  calo::multifit::ColumnVector<NPULSES>::Zero();

              // map views
              Eigen::Map<const calo::multifit::ColumnVector<NSAMPLES>> inputAmplitudesView{amplitudes + gch * NSAMPLES};
              Eigen::Map<const calo::multifit::ColumnVector<NSAMPLES>> noiseTermsView{noiseTerms + gch * NSAMPLES};
              Eigen::Map<const calo::multifit::ColumnVector<NSAMPLES>> noiseElectronicView{electronicNoiseTerms +
                                                                                           gch * NSAMPLES};
              Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixMView{
                  pulseMatricesM + gch * NSAMPLES * NPULSES};
              Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixPView{
                  pulseMatricesP + gch * NSAMPLES * NPULSES};
              Eigen::Map<const calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> glbPulseMatrixView{
                  pulseMatrices + gch * NSAMPLES * NPULSES};
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
              for (int iter = 1; iter < nMaxItersMin; iter++) {
                //float covarianceMatrixStorage[MapSymM<float, NSAMPLES>::total];
                // NOTE: only works when NSAMPLES == NPULSES
                // if does not hold -> slightly rearrange shared mem to still reuse
                // shared memory
                float* covarianceMatrixStorage = shrMatrixLFnnlsStorage;
                calo::multifit::MapSymM<float, NSAMPLES> covarianceMatrix{covarianceMatrixStorage};
                CMS_UNROLL_LOOP
                for (int counter = 0; counter < calo::multifit::MapSymM<float, NSAMPLES>::total; counter++)
                  covarianceMatrixStorage[counter] = (noisecorr != 0.f) ? 0.f : averagePedestalWidth2;
                CMS_UNROLL_LOOP
                for (unsigned int counter = 0; counter < calo::multifit::MapSymM<float, NSAMPLES>::stride; counter++) {
                  covarianceMatrix(counter, counter) += noiseTermsView.coeffRef(counter);
                  if (counter != 0)
                    covarianceMatrix(counter, counter - 1) +=
                        noisecorr * noiseElectronicView.coeffRef(counter - 1) * noiseElectronicView.coeffRef(counter);
                }

                // update covariance matrix
                update_covariance(resultAmplitudesVector,
                                  covarianceMatrix,
                                  glbPulseMatrixView,
                                  glbPulseMatrixMView,
                                  glbPulseMatrixPView);

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
                float matrixLStorage[calo::multifit::MapSymM<float, NSAMPLES>::total];
                calo::multifit::MapSymM<float, NSAMPLES> matrixL{matrixLStorage};
                calo::multifit::compute_decomposition_unrolled(matrixL, covarianceMatrix);

                //
                // replace eigen
                //
                //auto const& A = matrixDecomposition
                //    .matrixL()
                //    .solve(pulseMatrixView);
                calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES> A;
                calo::multifit::solve_forward_subst_matrix(A, glbPulseMatrixView, matrixL);

                //
                // remove eigen
                //
                //auto const& b = matrixL
                //   .solve(inputAmplitudesView);
                //
                float reg_b[NSAMPLES];
                calo::multifit::solve_forward_subst_vector(reg_b, inputAmplitudesView, matrixL);

                // TODO: we do not really need to change these matrcies
                // will be fixed in the optimized version
                //ColMajorMatrix<NPULSES, NPULSES> AtA = A.transpose() * A;
                //ColumnVector<NPULSES> Atb = A.transpose() * b;
                //ColMajorMatrix<NPULSES, NPULSES> AtA;
                //float AtAStorage[MapSymM<float, NPULSES>::total];
                calo::multifit::MapSymM<float, NPULSES> AtA{shrAtAStorage};
                calo::multifit::ColumnVector<NPULSES> Atb;
                CMS_UNROLL_LOOP
                for (int icol = 0; icol < NPULSES; icol++) {
                  float reg_ai[NSAMPLES];

                  // load column icol
                  CMS_UNROLL_LOOP
                  for (int counter = 0; counter < NSAMPLES; counter++)
                    reg_ai[counter] = A(counter, icol);

                  // compute diagonal
                  float sum = 0.f;
                  CMS_UNROLL_LOOP
                  for (int counter = 0; counter < NSAMPLES; counter++)
                    sum += reg_ai[counter] * reg_ai[counter];

                  // store
                  AtA(icol, icol) = sum;

                  // go thru the other columns
                  CMS_UNROLL_LOOP
                  for (int j = icol + 1; j < NPULSES; j++) {
                    // load column j
                    float reg_aj[NSAMPLES];
                    CMS_UNROLL_LOOP
                    for (int counter = 0; counter < NSAMPLES; counter++)
                      reg_aj[counter] = A(counter, j);

                    // accum
                    float sum = 0.f;
                    CMS_UNROLL_LOOP
                    for (int counter = 0; counter < NSAMPLES; counter++)
                      sum += reg_aj[counter] * reg_ai[counter];

                    // store
                    //AtA(icol, j) = sum;
                    AtA(j, icol) = sum;
                  }

                  // Atb accum
                  float sum_atb = 0;
                  CMS_UNROLL_LOOP
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
                calo::multifit::MapSymM<float, NPULSES> matrixLForFnnls{shrMatrixLFnnlsStorage};

                // run fast nnls
                calo::multifit::fnnls(AtA,
                                      Atb,
                                      resultAmplitudesVector,
                                      npassive,
                                      pulseOffsets,
                                      matrixLForFnnls,
                                      nnlsThresh,
                                      nMaxItersNNLS,
                                      10,
                                      10);

#ifdef HCAL_MAHI_GPUDEBUG
                printf("result Amplitudes\n");
                for (int i = 0; i < 8; i++)
                  printf("resultAmplitudes(%d) = %f\n", i, resultAmplitudesVector(i));
#endif

                calo::multifit::calculateChiSq(
                    matrixL, glbPulseMatrixView, resultAmplitudesVector, inputAmplitudesView, chi2);

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
                printf("pulseOffsets(%d) = %d outputAmplitudes(%d) = %f\n",
                       i,
                       pulseOffsets(i),
                       i,
                       resultAmplitudesVector(i));
              printf("chi2 = %f\n", chi2);
#endif

              outputGPU.chi2()[gch] = chi2;
              auto const idx_for_energy = std::abs(pulseOffsetsView.offsets()[0]);
              outputGPU.energy()[gch] = (gain * resultAmplitudesVector(idx_for_energy)) * respCorrection;

            }  // loop over channels
          }  //loop over group of channels
        }
      };  // Kernel_minimize

    }  // namespace mahi

    void runMahiAsync(Queue& queue,
                      IProductTypef01::ConstView const& f01HEDigis,
                      IProductTypef5::ConstView const& f5HBDigis,
                      IProductTypef3::ConstView const& f3HBDigis,
                      OProductType::View outputGPU,
                      HcalMahiConditionsPortableDevice::ConstView const& mahi,
                      HcalSiPMCharacteristicsPortableDevice::ConstView const& sipmCharacteristics,
                      HcalRecoParamWithPulseShapeDevice::ConstView const& recoParamsWithPS,
                      HcalMahiPulseOffsetsSoA::ConstView const& mahiPulseOffsets,
                      ConfigParameters const& configParameters) {
      auto const totalChannels =
          f01HEDigis.metadata().size() + f5HBDigis.metadata().size() + f3HBDigis.metadata().size();
      // FIXME: the number of channels in output might change given that some channesl might be filtered out

      // TODO: this can be lifted by implementing a separate kernel
      // similar to the default one, but properly handling the diff in #sample
      // or modifying existing one
      // TODO: assert startingSample = f01nsamples - windowSize to be 0 or 2
      // assert f01nsamples == f5nsamples
      // assert f01nsamples == f3nsamples
      int constexpr windowSize = 8;

      //compute work division
      uint32_t nchannels_per_block = configParameters.kprep1dChannelsPerBlock;
      auto const blocks_y = cms::alpakatools::divide_up_by(totalChannels, nchannels_per_block);

      Vec2D const blocks_2d{blocks_y, 1u};  // {y, x} coordiantes
      Vec2D const threads_2d{nchannels_per_block, windowSize};
      auto workDivPrep2D = cms::alpakatools::make_workdiv<Acc2D>(blocks_2d, threads_2d);

      //Device buffer for output
      auto amplitudes = cms::alpakatools::make_device_buffer<float[]>(queue, totalChannels * windowSize);
      auto noiseTerms = cms::alpakatools::make_device_buffer<float[]>(queue, totalChannels * windowSize);
      auto electronicNoiseTerms = cms::alpakatools::make_device_buffer<float[]>(queue, totalChannels * windowSize);
      auto soiSamples = cms::alpakatools::make_device_buffer<int8_t[]>(queue, totalChannels * windowSize);

      alpaka::exec<Acc2D>(queue,
                          workDivPrep2D,
                          mahi::Kernel_prep1d_sameNumberOfSamples{},
                          outputGPU,
                          f01HEDigis,
                          f5HBDigis,
                          f3HBDigis,
                          mahi,
                          sipmCharacteristics,
                          recoParamsWithPS,
                          configParameters.useEffectivePedestals,
                          configParameters.sipmQTSShift,
                          configParameters.sipmQNTStoSum,
                          configParameters.firstSampleShift,
                          configParameters.ts4Thresh,
                          amplitudes.data(),
                          noiseTerms.data(),
                          electronicNoiseTerms.data(),
                          soiSamples.data(),
                          windowSize);

      //// 1024 is the max threads per block for gtx1080
      //// FIXME: Take this from Alpaka in a way that does not need to query deviceProperty for every event
      uint32_t const channelsPerBlock = 1024 / (windowSize * mahiPulseOffsets.metadata().size());

      //launch 1D blocks of 3D threads
      auto const blocks_z = cms::alpakatools::divide_up_by(totalChannels, channelsPerBlock);
      Vec3D const blocks_3d{blocks_z, 1u, 1u};  // 1D block in z {z,y,x} coordinates
      Vec3D const threads_3d{channelsPerBlock, mahiPulseOffsets.metadata().size(), windowSize};

      auto workDivPrep3D = cms::alpakatools::make_workdiv<Acc3D>(blocks_3d, threads_3d);

      //Device buffer for output
      auto pulseMatrices = cms::alpakatools::make_device_buffer<float[]>(
          queue, totalChannels * windowSize * mahiPulseOffsets.metadata().size());
      auto pulseMatricesM = cms::alpakatools::make_device_buffer<float[]>(
          queue, totalChannels * windowSize * mahiPulseOffsets.metadata().size());
      auto pulseMatricesP = cms::alpakatools::make_device_buffer<float[]>(
          queue, totalChannels * windowSize * mahiPulseOffsets.metadata().size());

      alpaka::exec<Acc3D>(queue,
                          workDivPrep3D,
                          mahi::Kernel_prep_pulseMatrices_sameNumberOfSamples{},
                          pulseMatrices.data(),
                          pulseMatricesM.data(),
                          pulseMatricesP.data(),
                          mahiPulseOffsets,
                          amplitudes.data(),
                          f01HEDigis,
                          f5HBDigis,
                          f3HBDigis,
                          soiSamples.data(),
                          mahi,
                          recoParamsWithPS,
                          configParameters.meanTime,
                          configParameters.timeSigmaSiPM,
                          configParameters.timeSigmaHPD,
                          configParameters.applyTimeSlew,
                          configParameters.tzeroTimeSlew,
                          configParameters.slopeTimeSlew,
                          configParameters.tmaxTimeSlew);

      uint32_t threadsPerBlock = configParameters.kernelMinimizeThreads[0];
      auto blocks_1d = cms::alpakatools::divide_up_by(totalChannels, threadsPerBlock);

      auto workDivPrep1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_1d, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDivPrep1D,
                          mahi::Kernel_minimize<8, 8>{},
                          outputGPU,
                          amplitudes.data(),
                          pulseMatrices.data(),
                          pulseMatricesM.data(),
                          pulseMatricesP.data(),
                          mahiPulseOffsets,
                          noiseTerms.data(),
                          electronicNoiseTerms.data(),
                          soiSamples.data(),
                          mahi,
                          configParameters.useEffectivePedestals,
                          f01HEDigis,
                          f5HBDigis,
                          f3HBDigis);
    }

  }  // namespace hcal::reconstruction
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace alpaka::trait {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::hcal::reconstruction::mahi;

  //! The trait for getting the size of the block shared dynamic memory for Kernel_prep_1d_and_initialize.
  template <typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_prep1d_sameNumberOfSamples, TAcc> {
    //! \return The size of the shared memory allocated for a block.
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_prep1d_sameNumberOfSamples const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      // return the amount of dynamic shared memory needed

      // threadsPerBlock[1] = threads2d.x = windowSize = 8
      // threadsPerBlock[0] = threads2d.y = nchannels_per_block = 32
      // elemsPerThread = 1
      std::size_t bytes = threadsPerBlock[0u] * elemsPerThread[0u] *
                          ((2 * threadsPerBlock[1u] * elemsPerThread[1u] + 2) * sizeof(float) + sizeof(uint64_t));
      return bytes;
    }
  };

  //! The trait for getting the size of the block shared dynamic memory for kernel_minimize.
  template <int NSAMPLES, int NPULSES, typename TAcc>
  struct BlockSharedMemDynSizeBytes<Kernel_minimize<NSAMPLES, NPULSES>, TAcc> {
    //! \return The size of the shared memory allocated for a block.
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_minimize<NSAMPLES, NPULSES> const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      // return the amount of dynamic shared memory needed

      std::size_t bytes =
          2 * threadsPerBlock[0u] * elemsPerThread[0u] * (calo::multifit::MapSymM<float, 8>::total * sizeof(float));
      return bytes;
    }
  };

}  // namespace alpaka::trait
