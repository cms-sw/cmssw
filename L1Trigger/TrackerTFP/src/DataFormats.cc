#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
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

namespace trackerTFP {

  // default constructor, trying to need space as proper constructed object
  DataFormats::DataFormats()
      : numDataFormats_(0),
        formats_(+Variable::end, std::vector<DataFormat*>(+Process::end, nullptr)),
        numUnusedBitsStubs_(+Process::end, TTBV::S_ - 1),
        numUnusedBitsTracks_(+Process::end, TTBV::S_ - 1),
        numChannel_(+Process::end, 0) {
    setup_ = nullptr;
    countFormats();
    dataFormats_.reserve(numDataFormats_);
    numStreamsStubs_.reserve(+Process::end);
    numStreamsTracks_.reserve(+Process::end);
  }

  // method to count number of unique data formats
  template <Variable v, Process p>
  void DataFormats::countFormats() {
    if constexpr (config_[+v][+p] == p)
      numDataFormats_++;
    constexpr Process nextP = p + 1;
    if constexpr (nextP != Process::end)
      countFormats<v, nextP>();
    else {
      constexpr Variable nextV = v + 1;
      if constexpr (nextV != Variable::end)
        countFormats<nextV>();
    }
  }

  // proper constructor
  DataFormats::DataFormats(const tt::Setup* setup) : DataFormats() {
    setup_ = setup;
    fillDataFormats();
    for (const Process p : Processes)
      for (const Variable v : stubs_[+p])
        numUnusedBitsStubs_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
    for (const Process p : Processes)
      for (const Variable v : tracks_[+p])
        numUnusedBitsTracks_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
    numChannel_[+Process::dtc] = setup_->numDTCsPerRegion();
    numChannel_[+Process::pp] = setup_->numDTCsPerTFP();
    numChannel_[+Process::gp] = setup_->numSectors();
    numChannel_[+Process::ht] = setup_->htNumBinsInv2R();
    numChannel_[+Process::ctb] = setup_->kfNumWorker();
    numChannel_[+Process::kf] = setup_->kfNumWorker();
    numChannel_[+Process::dr] = 1;
    for (const Process& p : {Process::dtc, Process::pp, Process::gp, Process::ht}) {
      numStreamsStubs_.push_back(numChannel_[+p] * setup_->numRegions());
      numStreamsTracks_.push_back(0);
    }
    for (const Process& p : {Process::ctb, Process::kf, Process::dr}) {
      numStreamsTracks_.emplace_back(numChannel_[+p] * setup_->numRegions());
      numStreamsStubs_.emplace_back(numStreamsTracks_.back() * setup_->numLayers());
    }
  }

  // constructs data formats of all unique used variables and flavours
  template <Variable v, Process p>
  void DataFormats::fillDataFormats() {
    if constexpr (config_[+v][+p] == p) {
      dataFormats_.emplace_back(makeDataFormat<v, p>(setup_));
      fillFormats<v, p>();
    }
    constexpr Process nextP = p + 1;
    if constexpr (nextP != Process::end)
      fillDataFormats<v, nextP>();
    else {
      constexpr Variable nextV = v + 1;
      if constexpr (nextV != Variable::end)
        fillDataFormats<nextV>();
    }
  }

  // helper (loop) data formats of all unique used variables and flavours
  template <Variable v, Process p, Process it>
  void DataFormats::fillFormats() {
    if (config_[+v][+it] == p) {
      formats_[+v][+it] = &dataFormats_.back();
    }
    constexpr Process nextIt = it + 1;
    if constexpr (nextIt != Process::end)
      fillFormats<v, p, nextIt>();
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::tfp>(const tt::Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kRinvSize;
    const double range = -2. * TTTrack_TrackWord::minRinv;
    const double base = range * std::pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::tfp>(const tt::Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kPhiSize;
    const double range = -2. * TTTrack_TrackWord::minPhi0;
    const double base = range * std::pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tfp>(const tt::Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kTanlSize;
    const double range = -2. * TTTrack_TrackWord::minTanl;
    const double base = range * std::pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::tfp>(const tt::Setup* setup) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kZ0Size;
    const double range = -2. * TTTrack_TrackWord::minZ0;
    const double base = range * std::pow(2, -width);
    return DataFormat(true, width, base, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::r, Process::dtc>(const tt::Setup* setup) {
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::ht>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const int width = setup->tmttWidthR();
    const double range = 2. * setup->maxRphi();
    const double baseShifted = phiT.base() / inv2R.base();
    const int shift = std::ceil(std::log2(range / baseShifted)) - width;
    const double base = baseShifted * std::pow(2., shift);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::dtc>(const tt::Setup* setup) {
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::gp>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const int width = setup->tmttWidthPhi();
    const double range = phiT.range() + inv2R.range() * setup->maxRphi();
    const int shift = std::ceil(std::log2(range / phiT.base())) - width;
    const double base = phiT.base() * std::pow(2., shift);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z, Process::dtc>(const tt::Setup* setup) {
    const DataFormat zT = makeDataFormat<Variable::zT, Process::gp>(setup);
    const int width = setup->tmttWidthZ();
    const double range = 2. * setup->halfLength();
    const int shift = std::ceil(std::log2(range / zT.base())) - width;
    const double base = zT.base() * std::pow(2., shift);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::dtc>(const tt::Setup* setup) {
    const int width = 5;
    return DataFormat(false, width, 1., width);
  }

  template <>
  DataFormat makeDataFormat<Variable::phi, Process::gp>(const tt::Setup* setup) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::dtc>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::gp>(setup);
    const double base = phi.base();
    const double range = phiT.base() + inv2R.range() * setup->maxRphi();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z, Process::gp>(const tt::Setup* setup) {
    const DataFormat z = makeDataFormat<Variable::z, Process::dtc>(setup);
    const DataFormat zT = makeDataFormat<Variable::zT, Process::gp>(setup);
    const double rangeCot = (zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    const double base = z.base();
    const double range = zT.base() + rangeCot * setup->maxRz();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::gp>(const tt::Setup* setup) {
    const double range = 2. * M_PI / setup->numRegions();
    const int width = std::ceil(std::log2(setup->gpNumBinsPhiT()));
    const double base = range / std::pow(2., width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::gp>(const tt::Setup* setup) {
    const double range = 2. * std::sinh(setup->maxEta()) * setup->chosenRofZ();
    const double base = range / setup->gpNumBinsZT();
    const int width = std::ceil(std::log2(setup->gpNumBinsZT()));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::gp>(const tt::Setup* setup) {
    const DataFormat zT = makeDataFormat<Variable::zT, Process::gp>(setup);
    const DataFormat r = makeDataFormat<Variable::r, Process::dtc>(setup);
    const int width = setup->widthDSPbb();
    const double range = (zT.range() - zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    const double baseShifted = zT.base() / r.base();
    const int baseShift = std::ceil(std::log2(range / baseShifted)) - width;
    const double base = baseShifted * std::pow(2, baseShift);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::gp>(const tt::Setup* setup) {
    const int width = 6;
    return DataFormat(false, width, 1., width);
  }

  template <>
  DataFormat makeDataFormat<Variable::phi, Process::ht>(const tt::Setup* setup) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::dtc>(setup);
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::ht>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const double range = phiT.base() + setup->maxRphi() * inv2R.base();
    const double base = phi.base();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::ht>(const tt::Setup* setup) {
    const double range = 2. * setup->invPtToDphi() / setup->minPt();
    const double base = range / setup->htNumBinsInv2R();
    const int width = std::ceil(std::log2(setup->htNumBinsInv2R()));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::ht>(const tt::Setup* setup) {
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::gp>(setup);
    const double range = phiT.range();
    const double base = phiT.base() / setup->htNumBinsPhiT();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::dPhi, Process::ctb>(const tt::Setup* setup) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::dtc>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const double sigma = setup->pitchRowPS() / 2. / setup->innerRadius();
    const double pt = (setup->pitchCol2S() + setup->scattering()) / 2. * inv2R.range() / 2.;
    const double range = sigma + pt;
    const double base = phi.base();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::dZ, Process::ctb>(const tt::Setup* setup) {
    const DataFormat z = makeDataFormat<Variable::z, Process::dtc>(setup);
    const double range = setup->pitchCol2S() / 2. * std::sinh(setup->maxEta());
    const double base = z.base();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::layer, Process::ctb>(const tt::Setup* setup) {
    const double range = setup->numLayers();
    const int width = std::ceil(std::log2(range));
    return DataFormat(false, width, 1., range);
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::kf>(const tt::Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::inv2R, Process::tfp>(setup);
    const DataFormat ht = makeDataFormat<Variable::inv2R, Process::ht>(setup);
    const double range = ht.range() + 2. * ht.base();
    const double base = ht.base() * std::pow(2., std::floor(std::log2(.5 * tfp.base() / ht.base())));
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::kf>(const tt::Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::phiT, Process::tfp>(setup);
    const DataFormat ht = makeDataFormat<Variable::phiT, Process::ht>(setup);
    const double range = ht.range() + 2. * ht.base();
    const double base = ht.base() * std::pow(2., std::floor(std::log2(tfp.base() / ht.base())));
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::kf>(const tt::Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::cot, Process::tfp>(setup);
    const DataFormat zT = makeDataFormat<Variable::zT, Process::gp>(setup);
    const DataFormat r = makeDataFormat<Variable::r, Process::dtc>(setup);
    const double range = (zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    const double base = zT.base() / r.base() * std::pow(2., std::floor(std::log2(tfp.base() / zT.base() * r.base())));
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::kf>(const tt::Setup* setup) {
    const DataFormat tfp = makeDataFormat<Variable::zT, Process::tfp>(setup);
    const DataFormat gp = makeDataFormat<Variable::zT, Process::gp>(setup);
    const double range = gp.range();
    const double base = gp.base() * std::pow(2., std::floor(std::log2(tfp.base() / gp.base())));
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::kf>(const tt::Setup* setup) {
    const DataFormat phi = makeDataFormat<Variable::phi, Process::dtc>(setup);
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::kf>(setup);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::kf>(setup);
    const double range = phiT.base() + setup->maxRphi() * inv2R.base();
    const double base = phi.base();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::match, Process::kf>(const tt::Setup* setup) {
    const int width = 1;
    return DataFormat(false, width, 1., width);
  }

  template <>
  DataFormat makeDataFormat<Variable::cot, Process::dr>(const tt::Setup* setup) {
    const DataFormat kf = makeDataFormat<Variable::cot, Process::kf>(setup);
    const DataFormat zT = makeDataFormat<Variable::zT, Process::kf>(setup);
    const double range = (zT.range() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    const double base = kf.base();
    const int width = std::ceil(std::log2(range / base));
    return DataFormat(true, width, base, range);
  }

}  // namespace trackerTFP
