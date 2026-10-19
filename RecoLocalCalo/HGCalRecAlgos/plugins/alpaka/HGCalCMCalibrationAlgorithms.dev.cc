#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace F = hgcalcmml::Feature;  // This is actually just an index map defined in HGCALSoACMML.h

  static constexpr uint32_t kChPerErx = 37u;
  static constexpr float kCMScale = 2.0f;  // training convention: cm - 2*CM_ped

  // features 0..250, one thread per module; module m occupies digis [chDataOffsets[m], +nErx*37)
  struct HGCalCMCalibKernel_fillModuleInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoACMMLDeviceCollection::View module_features,
                                  hgcaldigi::HGCalDigiDevice::ConstView digi_view,
                                  uint32_t const* __restrict__ d_chDataOffsets,
                                  uint32_t const* __restrict__ d_enabledErx,
                                  int const* __restrict__ d_onehotIdx,
                                  unsigned char const* __restrict__ d_inScope,
                                  hgcalrechit::HGCalCalibParamDevice::ConstView calib,
                                  bool normalize,
                                  float const* d_inMean,
                                  float const* d_inInvStd,
                                  uint32_t nmodules,
                                  uint32_t ndigis) const {
      for (auto m : uniform_elements(acc, nmodules)) {
        auto f = module_features[m].features();
        if (!d_inScope[m]) {
          for (uint32_t k = 0; k < F::NMODULE; ++k)
            f(k) = 0.f;  // sets all the module feautures = 0
          continue;
        }
        uint32_t const base = d_chDataOffsets[m];

        // disabled or uncalibrated eRx are 0, very simple just fills in the CM.
        for (uint32_t e = 0; e < F::NCM; ++e) {
          uint32_t const src = base + e * kChPerErx;
          bool const on = ((d_enabledErx[m] >> e) & 1u) && src < ndigis && calib[src].valid();
          f(F::CM0 + e) = on ? kCMScale * (0.5f * float(digi_view[src].cm()) - calib[src].CM_ped()) : 0.f;
        }
        // Fills in TOA information and informaton relating to how many channels are above a given threshold.
        float ntoa = 0.f, ntot = 0.f;
        float nadcgt[F::NNCHADCGT] = {0.f, 0.f, 0.f, 0.f};
        constexpr float thresholds[F::NNCHADCGT] = {10.f, 50.f, 200.f, 500.f};  // namespace arrays not usable on device
        for (uint32_t i = 0; i < F::NCH; ++i) {
          uint32_t const di = base + i;
          float pedsub = 0.f;
          if (di < ndigis) {
            auto const digi = digi_view[di];
            if (digi.flags() != ::hgcal::DIGI_FLAG::NotAvailable) {
              ntoa += (digi.toa() > 0) ? 1.f : 0.f;
              ntot += (digi.tctp() == 3 && digi.tot() > 0) ? 1.f : 0.f;
            }
            pedsub = calib[di].valid() ? float(digi.adc()) - calib[di].ADC_ped() : 0.f;
          }
          for (uint32_t t = 0; t < F::NNCHADCGT; ++t)
            nadcgt[t] += (pedsub > thresholds[t]) ? 1.f : 0.f;
          f(F::ADC0 + i) = pedsub;
        }
        f(F::NCHTOA) = ntoa;
        f(F::NCHTOT) = ntot;
        for (uint32_t t = 0; t < F::NNCHADCGT; ++t)
          f(F::NCHADCGT0 + t) = nadcgt[t];
        for (uint32_t j = 0; j < F::NONEHOT; ++j)
          f(F::ONEHOT0 + j) = (d_onehotIdx[m] == int(j)) ? 1.f : 0.f;

        // For z score model we mean subtract and mutltiply by the inverse of the std.
        if (normalize) {
          for (uint32_t k = 0; k < F::NMODULE; ++k)
            f(k) = (f(k) - d_inMean[k]) * d_inInvStd[k];
        }
      }
    }
  };

  // one thread per digi: module row with the own ADC masked, plus the per-channel features
  struct HGCalCMCalibKernel_fillDigiInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoACMMLDeviceCollection::View digi_features,
                                  HGCalCMCorrectionDeviceCollection::View corrections,
                                  HGCalSoACMMLDeviceCollection::ConstView module_features,
                                  hgcal::HGCalDenseIndexInfoDevice::ConstView index_view,
                                  hgcal::HGCalMappingCellParamDevice::ConstView cellmap_view,
                                  uint32_t const* __restrict__ d_enabledErx,
                                  unsigned char const* __restrict__ d_inScope,
                                  float const* __restrict__ d_sfLD,
                                  float const* __restrict__ d_sfHD,
                                  bool normalize,
                                  float const* d_inMean,
                                  float const* d_inInvStd,
                                  uint32_t nmodules,
                                  uint32_t ndigis) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        // if DNN is not compatible for a channel zeros its inputs.
        auto const idxinfo = index_view[idx];
        uint32_t const m = idxinfo.modInfoIdx();
        uint32_t const chIdx = idxinfo.chNumber();
        auto f = digi_features[idx].features();
        if (m >= nmodules || !d_inScope[m] || chIdx >= F::NCH) {
          for (uint32_t k = 0; k < F::Size; ++k)
            f(k) = 0.f;
          corrections[idx].valid() = 0;
          continue;
        }

        auto const mf = module_features[m].features();
        for (uint32_t k = 0; k < F::NMODULE; ++k)
          f(k) = mf(k);
        uint32_t const own = F::ADC0 + chIdx;
        // sets its own digi adc to zero if usual DNN or the mean if z score DNN.
        f(own) = normalize ? -d_inMean[own] * d_inInvStd[own] : 0.f;

        // Channel and erx idx and always mean subtracted independent of the model.
        // They are then mean sub and std dev divided again if the z score model (a bit weird I know!)
        uint32_t const nErx = uint32_t(__builtin_popcount(d_enabledErx[m]));
        f(F::CHIDX) = float(chIdx) - (float(nErx) * float(kChPerErx) - 1.0f) / 2.0f;
        f(F::ERXIDX) = float(chIdx / kChPerErx) - (float(nErx) - 1.0f) / 2.0f;
        // We can clean this up with a new method added by p.silva but will probably merge this first.
        float const cellfrac = cellmap_view[idxinfo.cellInfoIdx()].isHD() ? d_sfHD[chIdx] : d_sfLD[chIdx];
        f(F::CELLFRAC) = cellfrac;
        // Z score model nornmalise.
        if (normalize) {
          for (uint32_t k = F::CHIDX; k < F::Size; ++k)
            f(k) = (f(k) - d_inMean[k]) * d_inInvStd[k];
        }
        // unconnected channels invalid.
        corrections[idx].valid() = (cellfrac != 0.f) ? 1 : 0;  // SF 0 = unconnected cell
      }
    }
  };

  // For the Z score DNN you need to un-normalise the output.
  struct HGCalCMCalibKernel_denormalizeOutputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalCMCorrectionDeviceCollection::View corrections,
                                  hgcal::HGCalDenseIndexInfoDevice::ConstView index_view,
                                  float const* __restrict__ d_tgtMean,
                                  float const* __restrict__ d_tgtStd,
                                  uint32_t ndigis) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        if (!corrections[idx].valid())
          continue;
        uint32_t const chIdx = index_view[idx].chNumber();
        corrections[idx].correction() = corrections[idx].correction() * d_tgtStd[chIdx] + d_tgtMean[chIdx];
      }
    }
  };

  void HGCalCMCalibrationAlgorithms::fillModuleInputs(Queue& queue,
                                                      uint32_t nmodules,
                                                      uint32_t ndigis,
                                                      hgcaldigi::HGCalDigiDevice const& device_digis,
                                                      uint32_t const* d_chDataOffsets,
                                                      uint32_t const* d_enabledErx,
                                                      int const* d_onehotIdx,
                                                      unsigned char const* d_inScope,
                                                      hgcalrechit::HGCalCalibParamDevice const& device_calib,
                                                      bool normalize,
                                                      float const* d_inMean,
                                                      float const* d_inInvStd,
                                                      HGCalSoACMMLDeviceCollection& device_module_features) const {
    if (nmodules == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(nmodules, 1),
                        HGCalCMCalibKernel_fillModuleInputs{},
                        device_module_features.view(),
                        device_digis.const_view(),
                        d_chDataOffsets,
                        d_enabledErx,
                        d_onehotIdx,
                        d_inScope,
                        device_calib.const_view(),
                        normalize,
                        d_inMean,
                        d_inInvStd,
                        nmodules,
                        ndigis);
  }

  void HGCalCMCalibrationAlgorithms::fillDigiInputs(Queue& queue,
                                                    uint32_t nmodules,
                                                    uint32_t ndigis,
                                                    hgcal::HGCalDenseIndexInfoDevice const& device_index,
                                                    hgcal::HGCalMappingCellParamDevice const& device_cellmap,
                                                    HGCalSoACMMLDeviceCollection const& device_module_features,
                                                    uint32_t const* d_enabledErx,
                                                    unsigned char const* d_inScope,
                                                    float const* d_sfLD,
                                                    float const* d_sfHD,
                                                    bool normalize,
                                                    float const* d_inMean,
                                                    float const* d_inInvStd,
                                                    HGCalSoACMMLDeviceCollection& device_digi_features,
                                                    HGCalCMCorrectionDeviceCollection& device_corrections) const {
    if (ndigis == 0)
      return;  // ha lets hope this doesn't happen, goodbye neutral hadrons.
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(ndigis, uint32_t(n_threads_)), uint32_t(n_threads_)),
                        HGCalCMCalibKernel_fillDigiInputs{},
                        device_digi_features.view(),
                        device_corrections.view(),
                        device_module_features.const_view(),
                        device_index.const_view(),
                        device_cellmap.const_view(),
                        d_enabledErx,
                        d_inScope,
                        d_sfLD,
                        d_sfHD,
                        normalize,
                        d_inMean,
                        d_inInvStd,
                        nmodules,
                        ndigis);
  }

  void HGCalCMCalibrationAlgorithms::denormalizeOutputs(Queue& queue,
                                                        uint32_t ndigis,
                                                        hgcal::HGCalDenseIndexInfoDevice const& device_index,
                                                        float const* d_tgtMean,
                                                        float const* d_tgtStd,
                                                        HGCalCMCorrectionDeviceCollection& device_corrections) const {
    if (ndigis == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(ndigis, uint32_t(n_threads_)), uint32_t(n_threads_)),
                        HGCalCMCalibKernel_denormalizeOutputs{},
                        device_corrections.view(),
                        device_index.const_view(),
                        d_tgtMean,
                        d_tgtStd,
                        ndigis);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
