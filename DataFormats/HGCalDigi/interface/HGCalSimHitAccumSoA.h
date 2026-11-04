#ifndef DataFormats_HGCalDigi_interface_HGCalSimHitAccumSoA_h
#define DataFormats_HGCalDigi_interface_HGCalSimHitAccumSoA_h

#include <Eigen/Core>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {

  constexpr int kTimeBins = 11;
  constexpr int kInTimeBin = (kTimeBins + 1) / 2;

  using TimedAccumulatorF = Eigen::Matrix<float, kTimeBins, 1>;
  using TimedROCDigiI = Eigen::Matrix<uint32_t, kTimeBins, 1>;

  GENERATE_SOA_LAYOUT(HGCalSimHitAccumSoALayout,
                      SOA_COLUMN(TimedAccumulatorF, sumE),
                      SOA_COLUMN(TimedAccumulatorF, sumExT))
  using HGCalSimHitAccumSoA = HGCalSimHitAccumSoALayout<>;

  GENERATE_SOA_LAYOUT(HGCalRealisticSimHitSoALayout,
                      SOA_COLUMN(TimedAccumulatorF, sumQ),
                      SOA_COLUMN(TimedAccumulatorF, avgTOA))
  using HGCalRealisticSimHitSoA = HGCalRealisticSimHitSoALayout<>;

  GENERATE_SOA_LAYOUT(HGCalROCDigisAccumSoALayout, SOA_COLUMN(TimedROCDigiI, word), SOA_COLUMN(TimedROCDigiI, status))
  using HGCalROCDigisAccumSoA = HGCalROCDigisAccumSoALayout<>;

}  // namespace hgcaldigi

#endif
