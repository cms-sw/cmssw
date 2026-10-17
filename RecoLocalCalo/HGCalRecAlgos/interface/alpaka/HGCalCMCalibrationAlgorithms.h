#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMML.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMMLDeviceCollection.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCorrectionDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalCMCalibrationAlgorithms {
  public:
    explicit HGCalCMCalibrationAlgorithms(int n_threads) : n_threads_(n_threads) {}

    // Fill ML input SoA (21 floats per digi) from digi + cell mapping + dense index info.
    // d_chDataOffsets[denseModIdx] = digi SoA offset of first channel for that module.
    // d_enabledErx[denseModIdx]   = number of active eRx for that module.
    // d_sfLD/d_sfHD: per-channel area scale factors for LD (222 entries) and HD (444 entries),
    //   indexed by chIdx (within-module channel number).
    // d_ntoa/d_ntot: per-module (indexed by denseModIdx) counts of TOA>0 / TOT-mode hits,
    //   computed host-side and broadcast to every SoA slot belonging to that module.
    void fillCMInputs(Queue& queue,
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
                      HGCalSoACMMLDeviceCollection& device_mlsoa) const;

    // Apply the per-channel subtractive correction (corrected = raw - prediction) in-place.
    // device_mlsoa carries the per-digi cellfrac (cell-area SF); the correction is skipped
    // for unconnected channels (cellfrac == 0). CM channels are not digi rows, so they are
    // never iterated here in the first place.
    void applyCMCorrections(Queue& queue,
                            uint32_t ndigis,
                            HGCalCMCorrectionDeviceCollection const& device_corrections,
                            HGCalSoACMMLDeviceCollection const& device_mlsoa,
                            hgcaldigi::HGCalDigiDevice& device_digis) const;

  private:
    int n_threads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCMCalibrationAlgorithms_h
