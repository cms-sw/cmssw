#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <string>
#include <iostream>
#include <numeric>

namespace trklet {

  // default constructor, trying to need space as proper constructed object
  DataFormats::DataFormats()
      : numDataFormats_(0),
        formats_(+Variable::end, std::vector<DataFormat*>(+Process::end, nullptr)),
        numUnusedBitsStubs_(+Process::end, TTBV::S_ - 1),
        numUnusedBitsTracks_(+Process::end, TTBV::S_ - 1) {
    setup_ = nullptr;
    countFormats();
    dataFormats_.reserve(numDataFormats_);
  }

  // method to count number of unique data formats
  template <Variable v, Process p>
  void DataFormats::countFormats() {
    if constexpr (config_[+v][+p] == p)
      numDataFormats_++;
    if constexpr (++p != Process::end)
      countFormats<v, ++p>();
    else if constexpr (++v != Variable::end)
      countFormats<++v>();
  }

  // proper constructor
  DataFormats::DataFormats(const Setup* setup) : DataFormats() {
    setup_ = setup;
    fillDataFormats();
    for (const Process p : Processes)
      for (const Variable v : stubs_[+p])
        numUnusedBitsStubs_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
    for (const Process p : Processes)
      for (const Variable v : tracks_[+p])
        numUnusedBitsTracks_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
  }

  // constructs data formats of all unique used variables and flavours
  template <Variable v, Process p>
  void DataFormats::fillDataFormats() {
    if constexpr (config_[+v][+p] == p) {
      dataFormats_.emplace_back(makeDataFormat<v, p>(setup_));
      fillFormats<v, p>();
    }
    if constexpr (++p != Process::end)
      fillDataFormats<v, ++p>();
    else if constexpr (++v != Variable::end)
      fillDataFormats<++v>();
  }

  // helper (loop) data formats of all unique used variables and flavours
  template <Variable v, Process p, Process it>
  void DataFormats::fillFormats() {
    if (config_[+v][+it] == p) {
      formats_[+v][+it] = &dataFormats_.back();
    }
    if constexpr (++it != Process::end)
      fillFormats<v, p, ++it>();
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::tfp>(const Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kRinvSize;
    const double range = -2. * TTTrack_TrackWord::minRinv;
    return DataFormat(true, width, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi0, Process::tfp>(const Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kPhiSize;
    const double range = -2. * TTTrack_TrackWord::minPhi0;
    return DataFormat(true, width, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tfp>(const Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kTanlSize;
    const double range = -2. * TTTrack_TrackWord::minTanl;
    return DataFormat(true, width, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z0, Process::tfp>(const Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kZ0Size;
    const double range = -2. * TTTrack_TrackWord::minZ0;
    return DataFormat(true, width, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::r, Process::dr>(const Setup* setup) {
    const DataFormat phi0 = makeDataFormat<Variable::phi0, Process::kf>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::kf>(setup);
    const double range = setup->sysOuterRadius();
    const double baseShifted = phi0.base() / inv2R.base();
    const int shift = tt::ilog2(range / baseShifted) - setup->drWidthR();
    const double base = baseShifted * std::pow(2., shift);
    return DataFormat(false, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::dr>(const Setup* setup) {
    const DataFormat phi0 = makeDataFormat<Variable::phi0, Process::kf>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::kf>(setup);
    const double range = setup->regRangePhiT() + setup->sysOuterRadius() * inv2R.range();
    const int shift = tt::ilog2(range / phi0.base()) - setup->drWidthPhi();
    const double base = phi0.base() * std::pow(2., shift);
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z, Process::dr>(const Setup* setup) {
    const DataFormat z0 = makeDataFormat<Variable::z0, Process::kf>(setup);
    const double range = 2. * setup->sysHalfLength();
    const int shift = tt::ilog2(range / z0.base()) - setup->drWidthZ();
    const double base = z0.base() * std::pow(2., shift);
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::dPhi, Process::dr>(const Setup* setup) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::dr>(setup);
    const int width = setup->drWidthDPhi();
    const int baseShift = setup->drBaseShiftDPhi();
    const double base = phi.base() * std::pow(2, baseShift);
    const double range = std::pow(2, width) * base;
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::dZ, Process::dr>(const Setup* setup) {
    const DataFormat z = makeDataFormat<Variable::z, Process::dr>(setup);
    const int width = setup->drWidthDZ();
    const int baseShift = setup->drBaseShiftDZ();
    const double base = z.base() * std::pow(2, baseShift);
    const double range = std::pow(2, width) * base;
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::seedType, Process::dr>(const Setup* setup) {
    const int width = tt::ilog2(setup->tbNumSeedTypes());
    return DataFormat(false, width);
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::kf>(const Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::inv2R, Process::tfp>(setup);
    const double range = 2. * setup->sysInvPtToDphi() / setup->regMinPt();
    const double base = range * std::pow(2., -tt::ilog2(range / tfp.base()) - 1);
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi0, Process::kf>(const Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::phi0, Process::tfp>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::kf>(setup);
    const double range = 2. * M_PI / setup->sysNumRegion() + setup->sysOuterRadius() * inv2R.range();
    const double base = range * std::pow(2., -tt::ilog2(range / tfp.base()) - 1);
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z0, Process::kf>(const Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::z0, Process::tfp>(setup);
    const double range = 2. * setup->regBeamWindowZ();
    const double base = range * std::pow(2., -tt::ilog2(range / tfp.base()) - 1);
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::kf>(const Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::cot, Process::tfp>(setup);
    const DataFormat z0 = makeDataFormat<Variable::z0, Process::kf>(setup);
    const DataFormat r = makeDataFormat<Variable::r, Process::dr>(setup);
    const double b = r.base() / z0.base();
    const double base = b * std::pow(2., -tt::ilog2(b / tfp.base()) - 1);
    const double rangeZT = 2. * std::sinh(setup->regMaxEta()) * setup->regChosenRofZ();
    const double range = (rangeZT + z0.range()) / setup->regChosenRofZ();
    return DataFormat(true, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::kf>(const Setup* setup) {
    const DataFormat dr = makeDataFormat<Variable::phi, Process::dr>(setup);
    const double base = dr.base() * std::pow(2, setup->kfBaseShiftPhi());
    const double range = dr.range();
    const int width = dr.width() - setup->kfBaseShiftPhi();
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z, Process::kf>(const Setup* setup) {
    const DataFormat dr = makeDataFormat<Variable::z, Process::dr>(setup);
    const double base = dr.base() * std::pow(2, setup->kfBaseShiftZ());
    const double range = dr.range();
    const int width = dr.width() - setup->kfBaseShiftPhi();
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::layerId, Process::kf>(const Setup* setup) {
    const int width = tt::ilog2(setup->kfNumLayers());
    return DataFormat(false, width);
  }

  template <>
  DataFormat makeDataFormat<Variable::chi20, Process::tq>(const Setup* setup) {
    const int shift = setup->tqBaseShiftChi20();
    const int width = setup->tqWidthChi20();
    const double base = std::pow(2., shift);
    const double range = base * std::pow(2, width);
    return DataFormat(false, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::chi21, Process::tq>(const Setup* setup) {
    const int shift = setup->tqBaseShiftChi21();
    const int width = setup->tqWidthChi21();
    const double base = std::pow(2., shift);
    const double range = base * std::pow(2, width);
    return DataFormat(false, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::mva, Process::tq>(const Setup* setup) {
    const int width = setup->tqWidthMVA();
    return DataFormat(false, width);
  }
  template <>
  DataFormat makeDataFormat<Variable::hitPattern, Process::tq>(const Setup* setup) {
    const int width = setup->kfNumLayers();
    return DataFormat(false, width);
  }

}  // namespace trklet
