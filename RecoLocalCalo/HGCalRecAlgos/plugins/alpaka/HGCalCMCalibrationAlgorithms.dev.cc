#include <cstddef>
#include <cstdint>
#include <algorithm>  // for std::clamp

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // ---------------------------------------------------------------------------
  // Kernel: fill CM calibration ML input SoA, one thread per digi.
  //
  // Indexing: digis are stored in global channel order, so digi_view[idx],
  // mlsoa[idx], and index_view[idx] all correspond to global channel idx.
  // The kernel iterates ndigis (the actual per-event digi count), not maxDataSize().
  //
  // DNN input order (21 floats, must match training):
  //   cm0..cm11 | msubchidx | msuberxidx | cellfrac | unconn0..3 | ntoa | ntot
  //
  // Correction is subtractive: corrected_adc = raw_adc - model_prediction.
  // ---------------------------------------------------------------------------
  struct HGCalCMCalibKernel_fillInputs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoACMMLDeviceCollection::View mlsoa,
                                  hgcaldigi::HGCalDigiDevice::ConstView digi_view,
                                  hgcal::HGCalDenseIndexInfoDevice::ConstView index_view,
                                  hgcal::HGCalMappingCellParamDevice::ConstView cellmap_view,
                                  uint32_t const* __restrict__ d_chDataOffsets,
                                  uint32_t const* __restrict__ d_enabledErx,
                                  float const* __restrict__ d_sfLD,
                                  float const* __restrict__ d_sfHD,
                                  float const* __restrict__ d_adcPed,
                                  float const* __restrict__ d_cmPed,
                                  int const* __restrict__ d_ntoa,
                                  int const* __restrict__ d_ntot,
                                  uint32_t ndigis,
                                  uint64_t event_num,
                                  uint64_t debug_event,
                                  uint32_t debug_module,
                                  uint32_t debug_max_ch) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        auto const idxinfo = index_view[idx];
        uint32_t const denseModIdx = idxinfo.modInfoIdx();
        uint32_t const chIdx = idxinfo.chNumber();  // 0..(nErx*37-1) within module
        // enabledErxMask is a bitmask: bit e set means eRx e is present.
        uint32_t const enabledErxMask = d_enabledErx[denseModIdx];
        // Count of enabled eRxs for mean-subtraction: __builtin_popcount works on device.
        uint32_t const nErx = uint32_t(__builtin_popcount(enabledErxMask));
        uint32_t const chOffset = d_chDataOffsets[denseModIdx];  // digi SoA offset for this module
        uint32_t const erxIdx = chIdx / 37u;

        auto slot = mlsoa[idx];

        // CHECK HOW ARNIE DID THE MEAN

        // Mean-subtracted channel index (unique per channel, referenced to module mean).
        slot.msubchidx() = float(chIdx) - (float(nErx) * 37.0f - 1.0f) / 2.0f;

        // Mean-subtracted eRx index (same for all channels in the same eRx).
        slot.msuberxidx() = float(erxIdx) - (float(nErx) - 1.0f) / 2.0f;

        // Cell area fraction: SF from cellareas.json, indexed by within-module chIdx.
        // isHD() selects the MH_F (HD, 444-entry) table; otherwise ML_F (LD, 222-entry).
        bool const isHD = cellmap_view[idxinfo.cellInfoIdx()].isHD();
        slot.cellfrac() = isHD ? d_sfHD[chIdx] : d_sfLD[chIdx];

        // Per-module scalars, identical for every channel in the same module.
        slot.ntoa() = float(d_ntoa[denseModIdx]);
        slot.ntot() = float(d_ntot[denseModIdx]);

        // CM sum per eRx (cm0..cm11): pedestal-subtracted as 0.5*digi.cm() - CM_ped,
        // matching the analytic formula in HGCalRecHitCalibrationAlgorithms.
        // CM_ped is read from the first channel of each eRx (chOffset + e*37).
        // Inactive eRx slots stay 0.
#define FILL_CM(e_) \
        slot.cm##e_() = ((enabledErxMask >> uint32_t(e_)) & 1u) \
                        ? (0.5f * float(digi_view[chOffset + uint32_t(e_) * 37u].cm()) \
                           - d_cmPed[chOffset + uint32_t(e_) * 37u]) \
                        : 0.0f
        FILL_CM(0);  FILL_CM(1);  FILL_CM(2);  FILL_CM(3);
        FILL_CM(4);  FILL_CM(5);  FILL_CM(6);  FILL_CM(7);
        FILL_CM(8);  FILL_CM(9);  FILL_CM(10); FILL_CM(11);
#undef FILL_CM

        // 4 individual unconnected channel ADCs in this digi's eRx, pedestal-subtracted.
        // Within-eRx positions: 8, 17, 19, 28 (same for all LD modules).
        // CELL AREA SHOULD be 0 for unconnected channels Arne.
        constexpr uint32_t kUnconn[4] = {8u, 17u, 19u, 28u};
        slot.unconn0() = float(digi_view[chOffset + erxIdx * 37u + kUnconn[0]].adc())
                         - d_adcPed[chOffset + erxIdx * 37u + kUnconn[0]];
        slot.unconn1() = float(digi_view[chOffset + erxIdx * 37u + kUnconn[1]].adc())
                         - d_adcPed[chOffset + erxIdx * 37u + kUnconn[1]];
        slot.unconn2() = float(digi_view[chOffset + erxIdx * 37u + kUnconn[2]].adc())
                         - d_adcPed[chOffset + erxIdx * 37u + kUnconn[2]];
        slot.unconn3() = float(digi_view[chOffset + erxIdx * 37u + kUnconn[3]].adc())
                         - d_adcPed[chOffset + erxIdx * 37u + kUnconn[3]];

        // Debug: print all 21 DNN input features for a specific event+module.
        // debug_event==0 disables; debug_event==UINT32_MAX matches any event.
        // debug_module==UINT32_MAX matches any module; debug_max_ch limits channels printed.
        // Sentinel: debug_event==UINT32_MAX (4294967295) means "any event".
        // Must use ~uint32_t(0u) not ~uint64_t(0u) — the config parameter is unsigned int.

        bool const dbg_ev  = (debug_event != 0u)
                             && (debug_event == uint64_t(~uint32_t(0u)) || event_num == debug_event);
        bool const dbg_mod = (debug_module == ~0u || denseModIdx == debug_module);
        if (dbg_ev && dbg_mod && chIdx < debug_max_ch) {
          float const adc_digi    = float(digi_view[idx].adc());
          float const unconn0_raw = float(digi_view[chOffset + erxIdx * 37u + kUnconn[0]].adc());
          printf("[CMCalib ev=%llu mod=%u ch=%u] "
                 "digi_adc=%.1f unconn0_raw=%.1f | "
                 "cm0=%.1f cm1=%.1f cm2=%.1f cm3=%.1f cm4=%.1f cm5=%.1f "
                 "cm6=%.1f cm7=%.1f cm8=%.1f cm9=%.1f cm10=%.1f cm11=%.1f | "
                 "msubchidx=%.2f msuberxidx=%.2f cellfrac=%.4f | "
                 "unconn0=%.1f unconn1=%.1f unconn2=%.1f unconn3=%.1f | "
                 "ntoa=%.1f ntot=%.1f\n",
                 (unsigned long long)event_num, denseModIdx, chIdx,
                 adc_digi, unconn0_raw,
                 slot.cm0(),    slot.cm1(),  slot.cm2(),  slot.cm3(),
                 slot.cm4(),    slot.cm5(),  slot.cm6(),  slot.cm7(),
                 slot.cm8(),    slot.cm9(),  slot.cm10(), slot.cm11(),
                 slot.msubchidx(), slot.msuberxidx(), slot.cellfrac(),
                 slot.unconn0(), slot.unconn1(), slot.unconn2(), slot.unconn3(),
                 slot.ntoa(), slot.ntot());
        }
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Kernel: apply subtractive DNN correction to digi ADC in-place.
  // corrected_adc = raw_adc - model_prediction  (model predicts the noise)
  //
  // Unconnected channels are left untouched: they carry no sensor cell, so their
  // cell-area SF (cellfrac, from cellareas.json) is 0 and the DNN prediction is
  // meaningless for them. CM channels are not part of the digi collection at all
  // (37 entries/eRx = 36 data + 1 calib; CM lives in the per-digi .cm() field),
  // so they are never iterated here.
  // ---------------------------------------------------------------------------
  struct HGCalCMCalibKernel_applyCorrections {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  hgcaldigi::HGCalDigiDevice::View digi_view,
                                  HGCalCMCorrectionDeviceCollection::ConstView corr_view,
                                  HGCalSoACMMLDeviceCollection::ConstView mlsoa,
                                  uint32_t ndigis) const {
      for (auto idx : uniform_elements(acc, ndigis)) {
        // Skip unconnected channels (cell-area SF == 0): leave the raw ADC as-is.
        if (mlsoa[idx].cellfrac() == 0.0f)
          continue;
        float corrected = float(digi_view[idx].adc()) - corr_view[idx].correction();
        digi_view[idx].adc() = uint16_t(std::clamp(corrected, 0.0f, 65535.0f));
      }
    }
  };

  // ---------------------------------------------------------------------------
  // fillCMInputs: fill ML input SoA from digis, one slot per digi.
  // ---------------------------------------------------------------------------
  void HGCalCMCalibrationAlgorithms::fillCMInputs(Queue& queue,
                                                   uint32_t ndigis,
                                                   int const* d_ntoa,
                                                   int const* d_ntot,
                                                   hgcaldigi::HGCalDigiDevice const& device_digis,
                                                   hgcal::HGCalDenseIndexInfoDevice const& device_index,
                                                   hgcal::HGCalMappingCellParamDevice const& device_cellmap,
                                                   uint32_t const* d_chDataOffsets,
                                                   uint32_t const* d_enabledErx,
                                                   float const* d_sfLD,
                                                   float const* d_sfHD,
                                                   float const* d_adcPed,
                                                   float const* d_cmPed,
                                                   uint64_t event_num,
                                                   uint64_t debug_event,
                                                   uint32_t debug_module,
                                                   uint32_t debug_max_ch,
                                                   HGCalSoACMMLDeviceCollection& device_mlsoa) const {
    LogDebug("HGCalCMCalibrationAlgorithms") << "fillCMInputs: ndigis=" << ndigis;

    uint32_t items = uint32_t(n_threads_);
    uint32_t groups = divide_up_by(ndigis, items);
    auto grid = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalCMCalibKernel_fillInputs{},
                        device_mlsoa.view(),
                        device_digis.const_view(),
                        device_index.const_view(),
                        device_cellmap.const_view(),
                        d_chDataOffsets,
                        d_enabledErx,
                        d_sfLD,
                        d_sfHD,
                        d_adcPed,
                        d_cmPed,
                        d_ntoa,
                        d_ntot,
                        ndigis,
                        event_num,
                        debug_event,
                        debug_module,
                        debug_max_ch);
  }

  // ---------------------------------------------------------------------------
  // applyCMCorrections: subtract DNN prediction from digi ADC in-place.
  // ---------------------------------------------------------------------------
  void HGCalCMCalibrationAlgorithms::applyCMCorrections(
      Queue& queue,
      uint32_t ndigis,
      HGCalCMCorrectionDeviceCollection const& device_corrections,
      HGCalSoACMMLDeviceCollection const& device_mlsoa,
      hgcaldigi::HGCalDigiDevice& device_digis) const {
    LogDebug("HGCalCMCalibrationAlgorithms") << "applyCMCorrections: ndigis=" << ndigis;

    uint32_t items = uint32_t(n_threads_);
    uint32_t groups = divide_up_by(ndigis, items);
    auto grid = make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalCMCalibKernel_applyCorrections{},
                        device_digis.view(),
                        device_corrections.const_view(),
                        device_mlsoa.const_view(),
                        ndigis);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
