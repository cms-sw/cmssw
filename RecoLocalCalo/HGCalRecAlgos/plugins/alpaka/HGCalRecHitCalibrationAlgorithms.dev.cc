#include <cstddef>

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// HGCal imports
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  //
  struct HGCalRecHitCalibrationKernel_flagRecHits {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        int calibvalid = std::to_integer<int>(calib.valid());
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        //recHits[idx].flags() = digiflags;
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         (calibvalid > 0));
        bool isToAavailable((digiflags != hgcal::DIGI_FLAG::ZS_ToA) && (digiflags != hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        recHits[idx].flags() = (!isAvailable) * hgcalrechit::HGCalRecHitFlags::EnergyInvalid +
                               (!isToAavailable) * hgcalrechit::HGCalRecHitFlags::TimeInvalid;
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_adcToCharge {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto adc_to_fC = [&](uint32_t adc,
                           uint32_t cm,
                           uint32_t adcm1,
                           float adc_ped,
                           float cm_slope,
                           float cm_ped,
                           float bxm1_slope,
                           float adc2fC) {
        return adc2fC * ((adc - adc_ped) - cm_slope * (cm - cm_ped) - bxm1_slope * (adcm1 - adc_ped));
      };

      auto tot_to_fC =
          [&](uint32_t tot, float tot_lin, float tot_ped, float tot2fC, float tot_p0, float tot_p1, float tot_p2) {
            bool isLin(tot > tot_lin);
            bool isNotLin(!isLin);
            return isLin * (tot2fC * (tot - tot_ped)) + isNotLin * (tot_p0 + tot_p1 * tot + tot_p2 * tot * tot);
          };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        int calibvalid = std::to_integer<int>(calib.valid());
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         (calibvalid > 0));
        bool useTOT((digi.tctp() == 3) && isAvailable);
        bool useADC(!useTOT && isAvailable);
        recHits[idx].energy() = useADC * adc_to_fC(digi.adc(),
                                                   digi.cm(),
                                                   digi.adcm1(),
                                                   calib.ADC_ped(),
                                                   calib.CM_slope(),
                                                   calib.CM_ped(),
                                                   calib.BXm1_slope(),
                                                   calib.ADCtofC()) +
                                useTOT * tot_to_fC(digi.tot(),
                                                   calib.TOT_lin(),
                                                   calib.TOTtofC(),
                                                   calib.TOT_ped(),
                                                   calib.TOT_P0(),
                                                   calib.TOT_P1(),
                                                   calib.TOT_P2());
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_toaToTime {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto toa_to_ps = [&](uint32_t toa, float toatops) { return toa * toatops; };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        int calibvalid = std::to_integer<int>(calib.valid());
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         (calibvalid > 0));
        bool isToAavailable((digiflags != hgcal::DIGI_FLAG::ZS_ToA) && (digiflags != hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        bool isGood(isAvailable && isToAavailable);
        recHits[idx].time() = isGood * toa_to_ps(digi.toa(), calib.TOAtops());
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_printRecHits {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalRecHitDevice::ConstView view, int size) const {
      for (int i = 0; i < size; ++i) {
        auto const& recHit = view[i];
        printf("%d\t%f\t%f\t%d\n", i, recHit.energy(), recHit.time(), recHit.flags());
      }
    }
  };

  std::unique_ptr<HGCalRecHitDevice> HGCalRecHitCalibrationAlgorithms::calibrate(
      Queue& queue,
      HGCalDigiHost const& host_digis,
      HGCalCalibParamDevice const& device_calib,
      HGCalConfigParamDevice const& device_config) const {
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Start of calibrate\n\n" << std::endl;
    LogDebug("HGCalRecHitCalibrationAlgorithms")
        << "N blocks: " << n_blocks_ << "\tN threads: " << n_threads_ << std::endl;
    auto grid = make_workdiv<Acc1D>(n_blocks_, n_threads_);

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Copying the digis to the device\n\n" << std::endl;
    HGCalDigiDevice device_digis(host_digis.view().metadata().size(), queue);
    alpaka::memcpy(queue, device_digis.buffer(), host_digis.const_buffer());

    LogDebug("HGCalRecHitCalibrationAlgorithms")
        << "\n\nINFO -- Allocating rechits buffer and initiating values" << std::endl;
    auto device_recHits = std::make_unique<HGCalRecHitDevice>(device_digis.view().metadata().size(), queue);

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_flagRecHits{},
                        device_digis.view(),
                        device_recHits->view(),
                        device_calib.view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_adcToCharge{},
                        device_digis.view(),
                        device_recHits->view(),
                        device_calib.view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_toaToTime{},
                        device_digis.view(),
                        device_recHits->view(),
                        device_calib.view());

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Input recHits: " << std::endl;
#ifdef EDM_ML_DEBUG
    int n_hits_to_print = 10;
    print_recHit_device(queue, *device_recHits, n_hits_to_print);
#endif

    return device_recHits;
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
                                                             HGCalRecHitDevice const& recHits,
                                                             int max) const {
    auto grid = make_workdiv<Acc1D>(1, 1);
    auto size = max > 0 ? max : recHits.view().metadata().size();
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_printRecHits{}, recHits.view(), size);

    // ensure that the print operations are complete before returning
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
