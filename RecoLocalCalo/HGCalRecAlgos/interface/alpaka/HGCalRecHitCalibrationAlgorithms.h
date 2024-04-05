#ifndef __HGCalRecHitCalibrationAlgorithms_H__
#define __HGCalRecHitCalibrationAlgorithms_H__

// CMSSW imports
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alpaka imports
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

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


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace hgcaldigi;
  using namespace hgcalrechit;

  class HGCalRecHitCalibrationAlgorithms {
  public:
    HGCalRecHitCalibrationAlgorithms(int n_blocks_, int n_threads_) : n_blocks(n_blocks_), n_threads(n_threads_) {}

    std::unique_ptr<HGCalRecHitDevice> calibrate(
      Queue& queue, HGCalDigiHost const& host_digis,
      HGCalCalibParamDevice const& device_calib, HGCalConfigParamDevice const& device_config
    );

    // if converting host digis to device rechits turns out too slow, we should copy host digis to device digis and then
    // convert to device rechits on device
    // std::unique_ptr<HGCalRecHitDevice> calibrate(Queue& queue, const std::unique_ptr<HGCalDigiDevice> &digis);

  private:
    void print(HGCalDigiHost const& digis, int max = -1) const;
    void print_digi_device(HGCalDigiDevice const& digis, int max = -1) const;
    void print_recHit_device(Queue& queue, HGCalRecHitDevice const& recHits, int max = -1) const;

    int n_blocks;
    int n_threads;

  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // __HGCalRecHitCalibrationAlgorithms_H__
