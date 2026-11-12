#ifndef DataFormats_HGCalDigi_interface_HGCalSimHitAccumSoA_h
#define DataFormats_HGCalDigi_interface_HGCalSimHitAccumSoA_h

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {

  constexpr int kTimeBins = 11;
  constexpr int kInTimeBin = 7;

  using TimedAccumulatorF = Eigen::Matrix<float, kTimeBins, 1>;
  using TimedROCDigiI = Eigen::Matrix<uint32_t, kTimeBins, 1>;

  GENERATE_SOA_LAYOUT(HGCalSimHitAccumSoALayout,
                      SOA_EIGEN_COLUMN(TimedAccumulatorF, sumE),
                      SOA_EIGEN_COLUMN(TimedAccumulatorF, sumExT))
  using HGCalSimHitAccumSoA = HGCalSimHitAccumSoALayout<>;

  GENERATE_SOA_LAYOUT(HGCalRealisticSimHitSoALayout,
                      SOA_EIGEN_COLUMN(TimedAccumulatorF, sumQ),
                      SOA_EIGEN_COLUMN(TimedAccumulatorF, avgTOA))
  using HGCalRealisticSimHitSoA = HGCalRealisticSimHitSoALayout<>;

  GENERATE_SOA_LAYOUT(HGCalPhaseShiftsSoALayout,
                      SOA_COLUMN(float, deltaT))
  using HGCalPhaseShiftsSoA = HGCalPhaseShiftsSoALayout<>;

  GENERATE_SOA_LAYOUT(HGCalROCDigisAccumSoALayout,
                      SOA_EIGEN_COLUMN(TimedROCDigiI, word),
                      SOA_EIGEN_COLUMN(TimedROCDigiI, status))
  using HGCalROCDigisAccumSoA = HGCalROCDigisAccumSoALayout<>;

  /**
  @short this is a helper class for validation purposes, it has no use for the digitization step itself
  it contains generic counters to check how many hits are rejected by time cuts or absense in mapping
  */
  struct HGCalSimHitAccumulatorMonitor {
    HGCalSimHitAccumulatorMonitor() : HGCalSimHitAccumulatorMonitor(0, false) {}
    HGCalSimHitAccumulatorMonitor(int16_t d, bool isc)
        : det(d),
          iscalib(isc),
          ncellsMapped(0),
          nHits(0),
          nHitsOOT(0),
          nHitsIT(0),
          nHitsRejected(0),
          nHitsUnknown(0),
          en(0.),
          enOOT(0.),
          enIT(0.),
          enRejected(0.),
          enUnknown(0.) {}
    HGCalSimHitAccumulatorMonitor(HGCalSimHitAccumulatorMonitor const&) = default;
    int16_t det;
    bool iscalib;
    int ncellsMapped;
    int nHits, nHitsOOT, nHitsIT, nHitsRejected, nHitsUnknown;
    float en, enOOT, enIT, enRejected, enUnknown;
  };
  typedef std::vector<HGCalSimHitAccumulatorMonitor> HGCalSimHitAccumulatorMonitorCollection;

}  // namespace hgcaldigi

#endif
