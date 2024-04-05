// Based on: https://github.com/CMS-HGCAL/cmssw/blob/hgcal-condformat-HGCalNANO-13_2_0_pre3_linearity/RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitCalibrationAlgorithms.dev.cc
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalRecHitCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  enum HGCalCalibrationFlag {
      kPedestalCorrection=0,
      kCMCorrection,
      kADCmCorrection,
  };

  class HGCalRecHitCalibrationKernel_digisToRecHits {
    public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits) const {
      auto ToA_to_time = [&](uint32_t ToA) { return float(ToA)*0.024414062; }; // LSB=25 ns / 2^10b
      auto ADC_to_float = [&](uint32_t adc, uint32_t tot, uint8_t tctp) { return float(tctp>1 ? tot : adc); };
      
      // dummy digis -> rechits conversion (to be replaced by the actual formula)
      for (auto idx : elements_with_stride(acc, digis.metadata().size())) {
        //recHits[idx].detid() = static_cast<uint32_t>(digis[idx].electronicsId()); // redundant since common dense indexing
        recHits[idx].energy() = ADC_to_float(digis[idx].adc(),digis[idx].tot(),digis[idx].tctp());
        recHits[idx].time() = ToA_to_time(digis[idx].toa());
        recHits[idx].flags() = digis[idx].flags();
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_pedestalCorrection {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits, HGCalCalibParamDevice::ConstView calib) const {
      for (auto idx : elements_with_stride(acc, digis.metadata().size())) {
        if ((digis[idx].tctp()==0) && (digis[idx].flags() >> kPedestalCorrection) & 1){
          recHits[idx].energy() = recHits[idx].energy() - calib[idx].ADC_ped();
        }
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_chargeConversion {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits, HGCalConfigParamDevice::ConstView config) const {
      auto ADC_to_charge = [&](float energy, uint8_t tctp, uint8_t gain) {
        return tctp>1 ? energy*1.953125 : energy*gain*0.078125; // fC
        //                TOT / 2^12 * 8000 fC = TOT * 1.953125 fC
        // ( ADC - pedestal ) / 2^10 *   80 fC = ( ADC - pedestal ) * 0.078125 fC
      };
      for (auto idx : elements_with_stride(acc, digis.metadata().size())) {
        recHits[idx].energy() = ADC_to_charge(recHits[idx].energy(),digis[idx].tctp(),config[idx].gain());
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_chargeConversion_exp {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits, HGCalConfigParamDevice::ConstView config) const {
      float calib_energy=0;
      auto ADC_to_charge = [&](float energy, uint8_t tctp, uint8_t gain) {
	      calib_energy =  tctp>1 ? energy*1.953125 : energy*gain*0.078125; 
        return calib_energy>0 ? calib_energy : 0.0; // fC
        //                TOT / 2^12 * 8000 fC = TOT * 1.953125 fC
        // ( ADC - pedestal ) / 2^10 *   80 fC = ( ADC - pedestal ) * 0.078125 fC
      };
      for (auto idx : elements_with_stride(acc, digis.metadata().size())) {
        recHits[idx].energy() = ADC_to_charge(recHits[idx].energy(),digis[idx].tctp(),config[idx].gain());
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_commonModeCorrection {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits, HGCalCalibParamDevice::ConstView calib) const {
      for (auto idx : elements_with_stride(acc, recHits.metadata().size())) {
        float commonModeValue = calib[idx].CM_slope() * ( digis[idx].cm() - calib[idx].CM_ped() );
        recHits[idx].energy() -= commonModeValue;
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_ADCmCorrection {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalDigiDevice::View digis, HGCalRecHitDevice::View recHits, HGCalCalibParamDevice::ConstView calib) const {
      for (auto idx : elements_with_stride(acc, recHits.metadata().size())) {
        float ADCmValue = calib[idx].BXm1_slope() * ( digis[idx].adcm1() - calib[idx].ADC_ped() ); // placeholder
        recHits[idx].adc() -= ADCmValue;
      }
    }
  };

  struct HGCalRecHitCalibrationKernel_printRecHits {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, HGCalRecHitDevice::ConstView view, int size) const {
      #ifdef EDM_ML_DEBUG
      for (int i = 0; i < size; ++i) {
        auto const& recHit = view[i];
        printf("%d\t%f\t%f\t%d\n", i, recHit.energy(), recHit.time(), recHit.flags());
      }
      #endif
    }
  };

  std::unique_ptr<HGCalRecHitDevice> HGCalRecHitCalibrationAlgorithms::calibrate(
      Queue& queue, HGCalDigiHost const& host_digis,
      HGCalCalibParamDevice const& device_calib, HGCalConfigParamDevice const& device_config
    ) {

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Start of calibrate\n\n" << std::endl;
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "N blocks: " << n_blocks << "\tN threads: " << n_threads << std::endl;
    auto grid = make_workdiv<Acc1D>(n_blocks, n_threads);

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Copying the digis to the device\n\n" << std::endl;
    HGCalDigiDevice device_digis(host_digis.view().metadata().size(), queue);
    alpaka::memcpy(queue, device_digis.buffer(), host_digis.const_buffer());

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "\n\nINFO -- Allocating rechits buffer and initiating values" << std::endl;
    auto device_recHits = std::make_unique<HGCalRecHitDevice>(device_digis.view().metadata().size(), queue);
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_digisToRecHits{}, device_digis.view(), device_recHits->view());
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Input recHits: " << std::endl;
    int n_hits_to_print = 10;
    print_recHit_device(queue, *device_recHits, n_hits_to_print);

    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_pedestalCorrection{}, device_digis.view(), device_recHits->view(), device_calib.view());
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "RecHits after pedestal calibration: " << std::endl;
    print_recHit_device(queue, *device_recHits, n_hits_to_print);

    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_commonModeCorrection{}, device_digis.view(), device_recHits->view(), device_calib.view());
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Digis after CM calibration: " << std::endl;
    //print_digi_device(device_digis, n_hits_to_print);
    print_recHit_device(queue, *device_recHits, n_hits_to_print);

    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_chargeConversion_exp{}, device_digis.view(), device_recHits->view(), device_config.view());
    //alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_chargeConversion{}, device_digis.view(), device_recHits->view(), device_config.view());
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "RecHits after charge conversion: " << std::endl;
    print_recHit_device(queue, *device_recHits, n_hits_to_print);

    /*
    float ADCmValue = n_hits_to_print; // dummy value
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_ADCmCorrection{}, device_digis.view(), calib_device.view());
    LogDebug("HGCalRecHitCalibrationAlgorithms") << "Digis after ADCm calibration: " << std::endl;
    print_digi_device(device_digis, n_hits_to_print);

    LogDebug("HGCalRecHitCalibrationAlgorithms") << "RecHits after calibration: " << std::endl;
    print_recHit_device(queue, *device_recHits, n_hits_to_print);
    */

    return device_recHits;
  }

  void HGCalRecHitCalibrationAlgorithms::print(HGCalDigiHost const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms") << i
        //<< "\t" << digis.view()[i].electronicsId()
        << "\t" << digis.view()[i].tctp()
        << "\t" << digis.view()[i].adcm1()
        << "\t" << digis.view()[i].adc()
        << "\t" << digis.view()[i].tot()
        << "\t" << digis.view()[i].toa()
        << "\t" << digis.view()[i].cm()
        << "\t" << digis.view()[i].flags()
        << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_digi_device(HGCalDigiDevice const& digis, int max) const {
    int max_ = max > 0 ? max : digis.view().metadata().size();
    for (int i = 0; i < max_; i++) {
      LogDebug("HGCalRecHitCalibrationAlgorithms") << i
        //<< "\t" << digis.view()[i].electronicsId()
        << "\t" << digis.view()[i].tctp()
        << "\t" << digis.view()[i].adcm1()
        << "\t" << digis.view()[i].adc()
        << "\t" << digis.view()[i].tot()
        << "\t" << digis.view()[i].toa()
        << "\t" << digis.view()[i].cm()
        << "\t" << digis.view()[i].flags()
        << std::endl;
    }
  }

  void HGCalRecHitCalibrationAlgorithms::print_recHit_device(Queue& queue, HGCalRecHitDevice const& recHits, int max) const {
    auto grid = make_workdiv<Acc1D>(1, 1);
    auto size = max > 0 ? max : recHits.view().metadata().size();
    alpaka::exec<Acc1D>(queue, grid, HGCalRecHitCalibrationKernel_printRecHits{}, recHits.view(), size);
    // ensure that the print operations are complete before returning
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

//
// Some potentially useful code snippets:
//

//void HGCalRecHitCalibrationAlgorithms::fill(Queue& queue, HGCalRecHitDevice& collection, double xvalue) const {
//  // use 64 items per group (this value is arbitrary, but it's a reasonable starting point)
//  uint32_t items = 64;

//  // use as many groups as needed to cover the whole problem
//  uint32_t groups = divide_up_by(collection->metadata().size(), items);

//  // map items to
//  // -threadswithasingleelementperthreadonaGPUbackend
//  // -elementswithinasinglethreadonaCPUbackend
//  auto workDiv = make_workdiv<Acc1D>(groups, items);

//  alpaka::exec<Acc1D>(queue, workDiv, HGCalRecHitCalibrationKernel{}, collection.view(), collection->metadata().size(), xvalue);
//}
