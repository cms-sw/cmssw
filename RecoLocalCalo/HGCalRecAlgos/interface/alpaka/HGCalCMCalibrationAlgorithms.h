#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMML.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMMLDeviceCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCorrectionDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Inputs of the multi-module CM DNN (layout in HGCALSoACMML.h). Normalisation arrays (z-score model only):
  // d_inMean/d_inInvStd have Feature::Size entries, d_tgtMean/d_tgtStd Feature::NCH; nullptr otherwise.
  class HGCalCMCalibrationAlgorithms {
  public:
    explicit HGCalCMCalibrationAlgorithms(int n_threads) : n_threads_(n_threads) {}

    // features 0..250, one row per module; ADC_ped/CM_ped/valid from the calibration, per dense channel index
    void fillModuleInputs(Queue& queue,
                          uint32_t nmodules,
                          uint32_t ndigis,
                          hgcaldigi::HGCalDigiDevice const& device_digis,
                          uint32_t const* d_chDataOffsets,
                          uint32_t const* d_enabledErx,
                          int const* d_onehotIdx,
                          unsigned char const* d_inScope,
                          hgcalrechit::HGCalCalibParamDevice const& device_calib, // if channel is invalid ped sub adc set to 0
                          bool normalize, // if true we normalise features, this is required for z score model, these come from two variables below!
                          float const* d_inMean,
                          float const* d_inInvStd,
                          HGCalSoACMMLDeviceCollection& device_module_features) const;

    // per digi: module row with the own ADC masked, per-channel features, device_corrections.valid
    void fillDigiInputs(Queue& queue,
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
                        HGCalCMCorrectionDeviceCollection& device_corrections) const;

    // z-score model: correction = prediction * d_tgtStd[chIdx] + d_tgtMean[chIdx]
    void denormalizeOutputs(Queue& queue,
                            uint32_t ndigis,
                            hgcal::HGCalDenseIndexInfoDevice const& device_index,
                            float const* d_tgtMean,
                            float const* d_tgtStd,
                            HGCalCMCorrectionDeviceCollection& device_corrections) const;

  private:
    int n_threads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
