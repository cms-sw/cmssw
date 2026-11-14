#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  namespace F = hgcalcmml::Feature;  // This is actually just an index map defined in HGCalSoACMML.h

  static constexpr uint32_t kChPerErx = 37u;
  static constexpr float kCMScale = 2.0f;  // training convention: cm - 2*CM_ped

  // features 0..250, one thread per module; module m occupies digis [chDataOffset, +nErx*37)
  struct HGCalCMCalibKernel_fillModuleInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoACMMLDeviceCollection::View module_features,
                                  hgcaldigi::HGCalDigiDevice::ConstView digi_view,
                                  HGCalCMMLModuleParamDevice::ConstView module_param,
                                  hgcalrechit::HGCalCalibParamDevice::ConstView calib,
                                  HGCalCMMLNormParamDevice::ConstView norm,
                                  uint32_t nmodules,
                                  uint32_t ndigis) const {
      for (auto m : uniform_elements(acc, nmodules)) {
        auto f = module_features[m].features();
        auto const mp = module_param[m];
        uint32_t const base = mp.chDataOffset();
        if (!mp.inScope() || base + F::NCH > ndigis) {
          for (uint32_t k = 0; k < F::NMODULE; ++k)
            f(k) = 0.f;  // sets all the module feautures = 0
          continue;
        }

        // disabled or uncalibrated eRx are 0, very simple just fills in the CM.
        for (uint32_t e = 0; e < F::NCM; ++e) {
          uint32_t const src = base + e * kChPerErx;
          bool const on = ((mp.enabledErx() >> e) & 1u) && src < ndigis && calib[src].valid();
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
          f(F::ONEHOT0 + j) = (mp.onehotIdx() == int(j)) ? 1.f : 0.f;

        // z-score: mean subtract and multiply by the inverse of the std.
        for (uint32_t k = 0; k < F::NMODULE; ++k)
          f(k) = (f(k) - norm[k].inMean()) * norm[k].inInvStd();
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
                                  HGCalCMMLModuleParamDevice::ConstView module_param,
                                  HGCalCMMLNormParamDevice::ConstView norm,
                                  uint32_t nmodules,
                                  uint32_t ndigis) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        // if DNN is not compatible for a channel zeros its inputs.
        auto const idxinfo = index_view[idx];
        uint32_t const m = idxinfo.modInfoIdx();
        uint32_t const chIdx = idxinfo.chNumber();
        auto f = digi_features[idx].features();
        if (m >= nmodules || !module_param[m].inScope() || module_param[m].chDataOffset() + F::NCH > ndigis ||
            chIdx >= F::NCH) {
          for (uint32_t k = 0; k < F::Size; ++k)
            f(k) = 0.f;
          corrections[idx].valid() = 0;
          continue;
        }

        auto const mf = module_features[m].features();
        for (uint32_t k = 0; k < F::NMODULE; ++k)
          f(k) = mf(k);
        uint32_t const own = F::ADC0 + chIdx;
        // sets its own digi adc to zero before the z-score, i.e. to the normalised mean.
        f(own) = -norm[own].inMean() * norm[own].inInvStd();

        // Channel and erx idx and always mean subtracted.
        // They are then mean sub and std dev divided again by the z-score (a bit weird I know!)
        uint32_t const nErx = uint32_t(__builtin_popcount(module_param[m].enabledErx()));
        f(F::CHIDX) = float(chIdx) - (float(nErx) * float(kChPerErx) - 1.0f) / 2.0f;
        f(F::ERXIDX) = float(chIdx / kChPerErx) - (float(nErx) - 1.0f) / 2.0f;
        // We can clean this up with a new method added by p.silva but will probably merge this first.
        float const cellfrac = module_param[m].cellFrac()(chIdx);
        f(F::CELLFRAC) = cellfrac;
        // z-score normalise.
        for (uint32_t k = F::CHIDX; k < F::Size; ++k)
          f(k) = (f(k) - norm[k].inMean()) * norm[k].inInvStd();
        // unconnected channels invalid.
        corrections[idx].valid() = (cellfrac != 0.f) ? 1 : 0;  // SF 0 = unconnected cell
      }
    }
  };

  // The z-score DNN predicts a normalised output, un-normalise it per channel.
  struct HGCalCMCalibKernel_denormalizeOutputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalCMCorrectionDeviceCollection::View corrections,
                                  hgcal::HGCalDenseIndexInfoDevice::ConstView index_view,
                                  HGCalCMMLNormParamDevice::ConstView norm,
                                  uint32_t ndigis) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        if (!corrections[idx].valid())
          continue;
        uint32_t const chIdx = index_view[idx].chNumber();
        corrections[idx].correction() = corrections[idx].correction() * norm[chIdx].tgtStd() + norm[chIdx].tgtMean();
      }
    }
  };

  void HGCalCMCalibrationAlgorithms::fillModuleInputs(Queue& queue,
                                                      uint32_t ndigis,
                                                      hgcaldigi::HGCalDigiDevice const& device_digis,
                                                      HGCalCMMLModuleParamDevice const& module_param,
                                                      hgcalrechit::HGCalCalibParamDevice const& device_calib,
                                                      HGCalCMMLNormParamDevice const& norm,
                                                      HGCalSoACMMLDeviceCollection& device_module_features) const {
    uint32_t const nmodules = module_param.const_view().metadata().size();
    if (nmodules == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(nmodules, 1),
                        HGCalCMCalibKernel_fillModuleInputs{},
                        device_module_features.view(),
                        device_digis.const_view(),
                        module_param.const_view(),
                        device_calib.const_view(),
                        norm.const_view(),
                        nmodules,
                        ndigis);
  }

  void HGCalCMCalibrationAlgorithms::fillDigiInputs(Queue& queue,
                                                    uint32_t ndigis,
                                                    hgcal::HGCalDenseIndexInfoDevice const& device_index,
                                                    HGCalSoACMMLDeviceCollection const& device_module_features,
                                                    HGCalCMMLModuleParamDevice const& module_param,
                                                    HGCalCMMLNormParamDevice const& norm,
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
                        module_param.const_view(),
                        norm.const_view(),
                        uint32_t(module_param.const_view().metadata().size()),
                        ndigis);
  }

  void HGCalCMCalibrationAlgorithms::denormalizeOutputs(Queue& queue,
                                                        uint32_t ndigis,
                                                        hgcal::HGCalDenseIndexInfoDevice const& device_index,
                                                        HGCalCMMLNormParamDevice const& norm,
                                                        HGCalCMCorrectionDeviceCollection& device_corrections) const {
    if (ndigis == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(ndigis, uint32_t(n_threads_)), uint32_t(n_threads_)),
                        HGCalCMCalibKernel_denormalizeOutputs{},
                        device_corrections.view(),
                        device_index.const_view(),
                        norm.const_view(),
                        ndigis);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
