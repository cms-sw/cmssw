#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalSoACMML.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMMLParamDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalSoACMMLDeviceCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCorrectionDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Inputs of the multi-module z-score CM DNN (layout in HGCalSoACMML.h). The per-module parameters and the
  // normalisation constants come from HGCalCMMLParamESProducer.
  class HGCalCMCalibrationAlgorithms {
  public:
    explicit HGCalCMCalibrationAlgorithms(int n_threads) : n_threads_(n_threads) {}

    // features 0..250, one row per module, z-scored; ADC_ped/CM_ped/valid from the calibration, per dense channel index
    void fillModuleInputs(Queue& queue,
                          uint32_t ndigis,
                          hgcaldigi::HGCalDigiDevice const& device_digis,
                          HGCalCMMLModuleParamDevice const& module_param,
                          hgcalrechit::HGCalCalibParamDevice const& device_calib,  // invalid channel: pedsub ADC = 0
                          HGCalCMMLNormParamDevice const& norm,
                          HGCalSoACMMLDeviceCollection& device_module_features) const;

    // per digi: module row with the own ADC masked, z-scored per-channel features, device_corrections.valid
    void fillDigiInputs(Queue& queue,
                        uint32_t ndigis,
                        hgcal::HGCalDenseIndexInfoDevice const& device_index,
                        HGCalSoACMMLDeviceCollection const& device_module_features,
                        HGCalCMMLModuleParamDevice const& module_param,
                        HGCalCMMLNormParamDevice const& norm,
                        HGCalSoACMMLDeviceCollection& device_digi_features,
                        HGCalCMCorrectionDeviceCollection& device_corrections) const;

    // correction = prediction * tgtStd[chIdx] + tgtMean[chIdx]
    void denormalizeOutputs(Queue& queue,
                            uint32_t ndigis,
                            hgcal::HGCalDenseIndexInfoDevice const& device_index,
                            HGCalCMMLNormParamDevice const& norm,
                            HGCalCMCorrectionDeviceCollection& device_corrections) const;

  private:
    int n_threads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
