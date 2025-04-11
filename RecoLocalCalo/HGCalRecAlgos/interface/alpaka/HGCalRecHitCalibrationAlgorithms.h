#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h

// Alpaka imports
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// HGCal digis data formats
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"

// Host & devide HGCal RecHit data formats
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalRecHit/interface/HGCalRecHitHost.h"
#include "DataFormats/HGCalRecHit/interface/alpaka/HGCalRecHitDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalCalibrationParameterDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace hgcaldigi;
  using namespace hgcalrechit;
  using namespace hgcal;

  class HGCalRecHitCalibrationAlgorithms {
  public:
    HGCalRecHitCalibrationAlgorithms(int n_blocks, int n_threads) : n_blocks_(n_blocks), n_threads_(n_threads) {}

    HGCalRecHitDevice calibrate(Queue& queue,
                                HGCalDigiHost const& host_digis,
                                HGCalCalibParamDevice const& device_calib,
                                HGCalConfigParamDevice const& device_config,
                                HGCalMappingCellParamDevice const& device_mapping,
                                HGCalDenseIndexInfoDevice const& device_index) const;

  private:
    void print(HGCalDigiHost const& digis, int max = -1) const;
    void print_digi_device(HGCalDigiDevice const& digis, int max = -1) const;
    void print_recHit_device(Queue& queue,
                             PortableHostCollection<hgcalrechit::HGCalRecHitSoALayout<> >::View const& recHits,
                             int max = -1) const;

    int n_blocks_;
    int n_threads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalRecHitCalibrationAlgorithms_h
