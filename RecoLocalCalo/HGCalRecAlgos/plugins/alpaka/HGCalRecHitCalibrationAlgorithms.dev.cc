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

  //
  struct HGCalRecHitCalibrationKernel_flagRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        //recHits[idx].flags() = digiflags;
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         calibvalid);
        bool isToAavailable((digiflags != hgcal::DIGI_FLAG::ZS_ToA) && (digiflags != hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        recHits[idx].flags() = (!isAvailable) * hgcalrechit::HGCalRecHitFlags::EnergyInvalid +
                               (!isToAavailable) * hgcalrechit::HGCalRecHitFlags::TimeInvalid;
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_adcToCharge {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
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
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         calibvalid);
        bool useTOT((digi.tctp() == 3) && isAvailable);
        bool useADC(!useTOT && isAvailable);
        recHits[idx].energy() = useADC * adc_denoise(digi.adc(),
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

        //after denoising/linearization apply the MIP scale
        recHits[idx].energy() *= calib.MIPS_scale();
      }
    }
  };

  //
  struct HGCalRecHitCalibrationKernel_toaToTime {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HGCalDigiDevice::View digis,
                                  HGCalRecHitDevice::View recHits,
                                  HGCalCalibParamDevice::ConstView calibs) const {
      auto toa_to_ps = [&](uint32_t toa, float toatops) { return toa * toatops; };

      for (auto idx : uniform_elements(acc, digis.metadata().size())) {
        auto calib = calibs[idx];
        bool calibvalid = calib.valid();
        auto digi = digis[idx];
        auto digiflags = digi.flags();
        bool isAvailable((digiflags != hgcal::DIGI_FLAG::Invalid) && (digiflags != hgcal::DIGI_FLAG::NotAvailable) &&
                         calibvalid);
        bool isToAavailable((digiflags != hgcal::DIGI_FLAG::ZS_ToA) && (digiflags != hgcal::DIGI_FLAG::ZS_ToA_ADCm1));
        bool isGood(isAvailable && isToAavailable);
        recHits[idx].time() = isGood * toa_to_ps(digi.toa(), calib.TOAtops());
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_printRecHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, HGCalRecHitDevice::ConstView view, int size) const {
      for (int i = 0; i < size; ++i) {
        auto const& recHit = view[i];
        printf("%d\t%f\t%f\t%d\n", i, recHit.energy(), recHit.time(), recHit.flags());
      }
    }
  };

  HGCalRecHitDevice HGCalRecHitCalibrationAlgorithms::calibrate(Queue& queue,
                                                                HGCalDigiHost const& host_digis,
                                                                HGCalCalibParamDevice const& device_calib,
                                                                HGCalConfigParamDevice const& device_config) const {
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Start of calibrate\n\n" << std::endl;

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Copying the digis to the device\n\n" << std::endl;
    HGCalDigiDevice device_digis(host_digis.view().metadata().size(), queue);
    alpaka::memcpy(queue, device_digis.buffer(), host_digis.const_buffer());

    LogDebug("HGCalRecHitCalibrationAlgorithms")
        << "\n\nINFO -- Allocating rechits buffer and initiating values" << std::endl;
    HGCalRecHitDevice device_recHits(device_digis.view().metadata().size(), queue);

    // number of items per group
    uint32_t items = n_threads_;
    // use as many groups as needed to cover the whole problem
    uint32_t groups = divide_up_by(device_digis.view().metadata().size(), items);
    // map items to
    //   - threads with a single element per thread on a GPU backend
    //   - elements within a single thread on a CPU backend
    auto grid = make_workdiv<Acc1D>(groups, items);
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "N groups: " << groups << "\tN items: " << items << std::endl;

    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_flagRecHits{},
                        device_digis.view(),
                        device_recHits.view(),
                        device_calib.view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_adcToCharge{},
                        device_digis.view(),
                        device_recHits.view(),
                        device_calib.view());
    alpaka::exec<Acc1D>(queue,
                        grid,
                        HGCalRecHitCalibrationKernel_toaToTime{},
                        device_digis.view(),
                        device_recHits.view(),
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

  void HGCalRecHitCalibrationAlgorithms::print_recHit_device(
      Queue& queue, PortableHostCollection<hgcalrechit::HGCalRecHitSoALayout<> >::View const& recHits, int max) const {
    auto grid = make_workdiv<Acc1D>(1, 1);
    auto size = max > 0 ? max : recHits.metadata().size();
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_printRecHits{}, recHits, size);

    // ensure that the print operations are complete before returning
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
