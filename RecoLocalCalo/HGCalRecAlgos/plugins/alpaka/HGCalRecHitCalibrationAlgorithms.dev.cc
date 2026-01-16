#include <cstddef>

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
//#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// HGCal imports
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // @short set RecHit flags based on DIGI flags (data corruption, zero-suppression)
  struct HGCalRecHitCalibrationKernel_flagRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        bool calibvalid = calibs[idx].valid();
        auto digiflags = digis[idx].flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool isToAavailable((digiflags != ::hgcal::DIGI_FLAG::ZS_ToA) &&
                            (digiflags != ::hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        recHits[idx].flags() =
            (!isAvailable) * HGCalRecHitFlags::kEnergyInvalid + (!isToAavailable) * HGCalRecHitFlags::kTimeInvalid;
      }
    }
  };

  // @short subtract pedestals, linearize ADC & TOT to charge (fC), and convert to energy (GeV)
  struct HGCalRecHitCalibrationKernel_adcToEnergy {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto adc_denoise =
          [&](uint32_t adc, uint32_t cm, uint32_t adcm1, float adc_ped, float cm_slope, float cm_ped, float bxm1_slope) {
            float cmf = cm_slope * (0.5 * float(cm) - cm_ped);
            return ((adc - adc_ped) - cmf - bxm1_slope * (adcm1 - adc_ped - cmf));
          };

      auto tot_linearization =
          [&](uint32_t tot, float tot_lin, float tot2adc, float tot_ped, float tot_p0, float tot_p1, float tot_p2) {
            bool isLin(tot > tot_lin);
            bool isNotLin(!isLin);
            return isLin * (tot2adc * (tot - tot_ped)) + isNotLin * (tot_p0 + tot_p1 * tot + tot_p2 * tot * tot);
          };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool useTOT((digi.tctp() == 3) && isAvailable);
        bool useADC(!useTOT && isAvailable);
        auto lin_counts = useADC * adc_denoise(digi.adc(),
                                               digi.cm(),
                                               digi.adcm1(),
                                               calib.ADC_ped(),
                                               calib.CM_slope(),
                                               calib.CM_ped(),
                                               calib.BXm1_slope()) +
                          useTOT * tot_linearization(digi.tot(),
                                                     calib.TOT_lin(),
                                                     calib.TOTtoADC(),
                                                     calib.TOT_ped(),
                                                     calib.TOT_P0(),
                                                     calib.TOT_P1(),
                                                     calib.TOT_P2());

        // after denoising/linearization apply the MIP & EM scale to convert to energy (GeV)
        // noise is also saved in units of GeV
        recHits[idx].mipEnergy() = lin_counts * calib.MIPS_scale();
        recHits[idx].energy() = calib.EM_scale() * calib.MIPS_scale() * lin_counts;
        recHits[idx].sigmaNoise() = calib.EM_scale() * calib.MIPS_scale() * calib.Noise();
      }
    }
  };

  // @short apply INL and timewalk corrections, and convert TOA to time (ps)
  struct HGCalRecHitCalibrationKernel_toaToTime {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto toa_inl_corr = [&](uint32_t toa, hgcalrechit::Vector32f ctdc_p, hgcalrechit::Vector8f ftdc_p) {
        auto gray = toa / 256;
        auto ptdc = toa % 256;
        auto ctdc = ptdc / 8;
        auto ftdc = ptdc % 8;
        auto ctdc_corr = uint(ctdc - ctdc_p[ctdc]) % 32;
        auto ftdc_corr = uint(ftdc - ftdc_p[ftdc]) % 8;
        return (ftdc_corr + 8 * ctdc_corr + 256 * gray) % 1024;
      };

      auto toa_tw_corr = [&](uint32_t toa, float energy, hgcalrechit::Vector3f p) {
        return toa - ((energy > p[2]) ? (p[0] + p[1] * std::log(energy - p[2])) : 0.f);
      };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool isToAavailable((digiflags != ::hgcal::DIGI_FLAG::ZS_ToA) &&
                            (digiflags != ::hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        bool isGood(isAvailable && isToAavailable);
        // INL correction
        auto toa = isGood * toa_inl_corr(digi.toa(), calib.TOA_CTDC(), calib.TOA_FTDC());
        // timewalk correction
        toa = isGood * toa_tw_corr(toa, recHits[idx].energy(), calib.TOA_TW());
        // toa to ps
        recHits[idx].time() = toa * hgcalrechit::TOAtops;
        recHits[idx].timeError() = 0.;
      }
    }
  };

  // @short sum the energy of the RecHits in the calibration and surrounding cells
  // see https://github.com/cms-sw/cmssw/pull/47829
  struct HGCalRecHitCalibrationKernel_handleCalibCell {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalDigiDevice::ConstView digis,
                                  HGCalCalibParamDevice::ConstView calibs,
                                  HGCalMappingCellParamDevice::ConstView maps,
                                  HGCalDenseIndexInfoDevice::ConstView index) const {
      auto time_average = [&](float time_surr, float time_calib, float energy_surr, float energy_calib) {
        bool is_time_surr(time_surr > 0);
        bool is_time_calib(time_calib > 0);
        float totalEn = (is_time_surr * energy_surr + is_time_calib * energy_calib);
        float weighted_average =
            (totalEn > 0)
                ? (is_time_surr * energy_surr * time_surr + is_time_calib * energy_calib * time_calib) / totalEn
                : 0.0f;
        return weighted_average;
      };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != ::hgcal::DIGI_FLAG::Invalid) &&
                         (digiflags != ::hgcal::DIGI_FLAG::NotAvailable) && calibvalid);
        bool isToAavailable((digiflags != ::hgcal::DIGI_FLAG::ZS_ToA) &&
                            (digiflags != ::hgcal::DIGI_FLAG::ZS_ToA_ADCm1));

        auto cellIndex = index[idx].cellInfoIdx();
        bool isCalibCell(maps[cellIndex].iscalib());
        int offset = maps[cellIndex].caliboffset();  // calibration-to-surrounding cell offset
        bool is_surr_cell((offset != 0) && isAvailable && isCalibCell);

        // effectively operate only on the cell that surrounds the calibration cells
        if (!is_surr_cell) {
          continue;
        }

        recHits[idx + offset].flags() = HGCalRecHitFlags::kNormal;

        recHits[idx + offset].time() = isToAavailable * time_average(recHits[idx + offset].time(),
                                                                     recHits[idx].time(),
                                                                     recHits[idx + offset].energy(),
                                                                     recHits[idx].energy());

        bool is_negative_surr_energy(recHits[idx + offset].energy() < 0);
        auto negative_energy_correction = (-1.0 * recHits[idx + offset].energy()) * is_negative_surr_energy;
        auto negative_mipEnergy_correction = (-1.0 * recHits[idx + offset].mipEnergy()) * is_negative_surr_energy;

        recHits[idx + offset].energy() += (negative_energy_correction + recHits[idx].energy());
        recHits[idx + offset].mipEnergy() += (negative_mipEnergy_correction + recHits[idx].mipEnergy());
        float quadratic_noise_sum = sqrt(recHits[idx].sigmaNoise() * recHits[idx].sigmaNoise() +
                                         recHits[idx + offset].sigmaNoise() * recHits[idx + offset].sigmaNoise());
        recHits[idx + offset].sigmaNoise() = quadratic_noise_sum;
      }
    }
  };

  // add metadata to rechits
  struct HGCalRecHitCalibrationKernel_metaData {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View recHits,
                                  HGCalMappingModuleParamDevice::ConstView mapmod,
                                  HGCalDenseIndexInfoDevice::ConstView index) const {
      for (auto idx : uniform_elements(acc, recHits.metadata().size())) {
        auto denseIdx = index[idx];
        auto recHit = recHits[idx];
        if (mapmod[denseIdx.modInfoIdx()].isSiPM()) {
          recHit.dim1() = denseIdx.eta();
          recHit.dim2() = denseIdx.phi();
        }  // else, isSilicon == true and eta phi values will not be used
        else {
          recHit.dim1() = denseIdx.x();
          recHit.dim2() = denseIdx.y();
        }
        recHit.dim3() = denseIdx.z();
        recHit.recHitIndex() = idx;
        recHit.detid() = denseIdx.detid();
        recHit.layer() = denseIdx.layer();
      }
    }
  };

  // @short print RecHits for debugging
  struct HGCalRecHitCalibrationKernel_printRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, HGCalSoARecHitsDeviceCollection::ConstView view, int size) const {
      for (int i = 0; i < size; ++i) {
        auto const& recHit = view[i];
        printf(
            "i=%d\t dim1=%f\t dim2=%f\t dim3=%f\t sigmaNoise=%f\t layer=%d\t detid=%u\t mipEnergy=%f\t "
            "energy=%f\t time=%f\t timeError=%f\t flags=%d\n",
            i,
            recHit.dim1(),
            recHit.dim2(),
            recHit.dim3(),
            recHit.sigmaNoise(),
            recHit.layer(),
            recHit.detid(),
            recHit.mipEnergy(),
            recHit.energy(),
            recHit.time(),
            recHit.timeError(),
            recHit.flags());
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_countRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  int32_t* __restrict__ nsel,
                                  int32_t* __restrict__ sidx,
                                  HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                  double k_noise = 0.) const {
      for (auto idx : uniform_elements(acc, recHits.metadata().size())) {
        if (!recHits[idx].flags() && recHits[idx].energy() > k_noise * recHits[idx].sigmaNoise() &&
            recHits[idx].layer() != 0)
          sidx[alpaka::atomicAdd(acc, nsel, 1)] = idx;
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_copyRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalSoARecHitsDeviceCollection::View output,
                                  HGCalSoARecHitsDeviceCollection::ConstView recHits,
                                  int32_t const* __restrict__ sidx) const {
      for (auto idx : uniform_elements(acc, output.metadata().size()))
        output[idx] = recHits[sidx[idx]];
    }
  };

  // @short apply all calibration kernels
  HGCalSoARecHitsDeviceCollection HGCalRecHitCalibrationAlgorithms::calibrate(
      Queue& queue,
      int32_t* __restrict__ nsel,
      int32_t* __restrict__ sidx,
      HGCalDigiHost const& host_digis,
      HGCalCalibParamDevice const& device_calib,
      HGCalMappingModuleParamDevice const& device_mapmod,
      HGCalMappingCellParamDevice const& device_mapping,
      HGCalDenseIndexInfoDevice const& device_index,
      double k_noise) const {
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Start of calibrate\n\n" << std::endl;

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Copying the digis to the device\n\n" << std::endl;
    auto const ndigis = host_digis.view().metadata().size();
    HGCalDigiDevice device_digis(ndigis, queue);
    alpaka::memcpy(queue, device_digis.buffer(), host_digis.const_buffer());

    LogDebug("HGCalRecHitCalibrationAlgorithms")
        << "\n\nINFO -- Allocating rechits buffer and initiating values" << std::endl;
    HGCalSoARecHitsDeviceCollection device_recHits(ndigis, queue);

    // number of items per group
    uint32_t items = n_threads_;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(ndigis, items);
    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_flagRecHits{},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_adcToEnergy{},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_toaToTime{},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_handleCalibCell{},
                        device_recHits.view(),
                        device_digis.const_view(),
                        device_calib.const_view(),
                        device_mapping.const_view(),
                        device_index.const_view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_metaData{},
                        device_recHits.view(),
                        device_mapmod.const_view(),
                        device_index.const_view());

    // select rec hits
    alpaka::exec<Acc1D>(
        queue, grid, HGCalRecHitCalibrationKernel_countRecHits{}, nsel, sidx, device_recHits.const_view(), k_noise);

    return device_recHits;
  }

  // @short select rechits
  HGCalSoARecHitsDeviceCollection HGCalRecHitCalibrationAlgorithms::select(
      Queue& queue,
      int const ndigis,
      int32_t const* __restrict__ nsel,
      int32_t const* __restrict__ sidx,
      HGCalSoARecHitsDeviceCollection const& device_recHits) const {
    // number of items per group
    uint32_t items = n_threads_;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(ndigis, items);
    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);

    HGCalSoARecHitsDeviceCollection device_selRecHits(*nsel, queue);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_copyRecHits{},
                        device_selRecHits.view(),
                        device_recHits.const_view(),
                        sidx);

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Input recHits: " << std::endl;
#ifdef EDM_ML_DEBUG
    int n_hits_to_print = 10;
    print_recHit_device(queue, *device_selRecHits, n_hits_to_print);
#endif

    return device_selRecHits;
  }

  void HGCalRecHitCalibrationAlgorithms::print(HGCalDigiHost const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms")
          << i << digis.view()[i].tot() << "\t" << digis.view()[i].toa() << "\t" << digis.view()[i].cm() << "\t"
          << digis.view()[i].flags() << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_digi_device(HGCalDigiDevice const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms")
          << i << digis.view()[i].tot() << "\t" << digis.view()[i].toa() << "\t" << digis.view()[i].cm() << "\t"
          << digis.view()[i].flags() << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_recHit_device(Queue& queue,
                                                             HGCalSoARecHitsHostCollection::View const& recHits,
                                                             int max) const {
    auto grid = make_workdiv<Acc1D>(1, 1);
    auto size = max > 0 ? max : recHits.metadata().size();
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_printRecHits{}, recHits, size);

    // ensure that the print operations are complete before returning
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
