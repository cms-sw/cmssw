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

using namespace std;
using namespace edm;
using namespace tt;

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
  DataFormats::DataFormats(const Setup* setup) : DataFormats() {
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
      dataFormats_.emplace_back(Format<v, p>(setup_));
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
  Format<Variable::inv2R, Process::tfp>::Format(const Setup* setup) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kRinvSize;
    range_ = -2. * TTTrack_TrackWord::minRinv;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::phiT, Process::tfp>::Format(const Setup* setup) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kPhiSize;
    range_ = -2. * TTTrack_TrackWord::minPhi0;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::cot, Process::tfp>::Format(const Setup* setup) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kTanlSize;
    range_ = -2. * TTTrack_TrackWord::minTanl;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::zT, Process::tfp>::Format(const Setup* setup) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kZ0Size;
    range_ = -2. * TTTrack_TrackWord::minZ0;
    base_ = range_ * pow(2, -width_);
  }

  template <>
  Format<Variable::r, Process::dtc>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::ht> phiT(setup);
    const Format<Variable::inv2R, Process::ht> inv2R(setup);
    width_ = setup->tmttWidthR();
    range_ = 2. * setup->maxRphi();
    base_ = phiT.base() / inv2R.base();
    const int shift = ceil(log2(range_ / base_)) - width_;
    base_ *= pow(2., shift);
  }
  template <>
  Format<Variable::phi, Process::dtc>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::gp> phiT(setup);
    const Format<Variable::inv2R, Process::ht> inv2R(setup);
    width_ = setup->tmttWidthPhi();
    range_ = phiT.range() + inv2R.range() * setup->maxRphi();
    const int shift = ceil(log2(range_ / phiT.base())) - width_;
    base_ = phiT.base() * pow(2., shift);
  }
  template <>
  Format<Variable::z, Process::dtc>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::zT, Process::gp> zT(setup);
    width_ = setup->tmttWidthZ();
    range_ = 2. * setup->halfLength();
    const int shift = ceil(log2(range_ / zT.base())) - width_;
    base_ = zT.base() * pow(2., shift);
  }
  template <>
  Format<Variable::layer, Process::dtc>::Format(const Setup* setup) : DataFormat(false) {
    width_ = 5;
  }

  template <>
  Format<Variable::phi, Process::gp>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phi, Process::dtc> phi(setup);
    const Format<Variable::inv2R, Process::ht> inv2R(setup);
    const Format<Variable::phiT, Process::gp> phiT(setup);
    base_ = phi.base();
    range_ = phiT.base() + inv2R.range() * setup->maxRphi();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::z, Process::gp>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::z, Process::dtc> z(setup);
    const Format<Variable::zT, Process::gp> zT(setup);
    const double rangeCot = (zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    base_ = z.base();
    range_ = zT.base() + rangeCot * setup->maxRz();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phiT, Process::gp>::Format(const Setup* setup) : DataFormat(true) {
    range_ = 2. * M_PI / setup->numRegions();
    width_ = ceil(log2(setup->gpNumBinsPhiT()));
    base_ = range_ / pow(2., width_);
  }
  template <>
  Format<Variable::zT, Process::gp>::Format(const Setup* setup) : DataFormat(true) {
    range_ = 2. * sinh(setup->maxEta()) * setup->chosenRofZ();
    base_ = range_ / setup->gpNumBinsZT();
    width_ = ceil(log2(setup->gpNumBinsZT()));
  }
  template <>
  Format<Variable::cot, Process::gp>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::zT, Process::gp> zT(setup);
    const Format<Variable::r, Process::dtc> r(setup);
    width_ = setup->widthDSPbb();
    range_ = (zT.range() - zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    base_ = zT.base() / r.base();
    const int baseShift = ceil(log2(range_ / base_)) - width_;
    base_ *= pow(2, baseShift);
  }
  template <>
  Format<Variable::layer, Process::gp>::Format(const Setup* setup) : DataFormat(false) {
    width_ = 6;
  }

  template <>
  Format<Variable::phi, Process::ht>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phi, Process::dtc> phi(setup);
    const Format<Variable::phiT, Process::ht> phiT(setup);
    const Format<Variable::inv2R, Process::ht> inv2R(setup);
    range_ = phiT.base() + setup->maxRphi() * inv2R.base();
    base_ = phi.base();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::inv2R, Process::ht>::Format(const Setup* setup) : DataFormat(true) {
    range_ = 2. * setup->invPtToDphi() / setup->minPt();
    base_ = range_ / (double)setup->htNumBinsInv2R();
    width_ = ceil(log2(setup->htNumBinsInv2R()));
  }
  template <>
  Format<Variable::phiT, Process::ht>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::gp> phiT(setup);
    range_ = phiT.range();
    base_ = phiT.base() / (double)setup->htNumBinsPhiT();
    width_ = ceil(log2(range_ / base_));
  }

  template <>
  Format<Variable::dPhi, Process::ctb>::Format(const Setup* setup) : DataFormat(false) {
    const Format<Variable::phi, Process::dtc> phi(setup);
    const Format<Variable::inv2R, Process::ht> inv2R(setup);
    const double sigma = setup->pitchRowPS() / 2. / setup->innerRadius();
    const double pt = (setup->pitchCol2S() + setup->scattering()) / 2. * inv2R.range() / 2.;
    range_ = sigma + pt;
    base_ = phi.base();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::dZ, Process::ctb>::Format(const Setup* setup) : DataFormat(false) {
    const Format<Variable::z, Process::dtc> z(setup);
    range_ = setup->pitchCol2S() / 2. * sinh(setup->maxEta());
    base_ = z.base();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::layer, Process::ctb>::Format(const Setup* setup) : DataFormat(false) {
    range_ = setup->numLayers();
    width_ = ceil(log2(range_));
  }

  template <>
  Format<Variable::inv2R, Process::kf>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::inv2R, Process::tfp> tfp(setup);
    const Format<Variable::inv2R, Process::ht> ht(setup);
    range_ = ht.range() + 2. * ht.base();
    base_ = ht.base() * pow(2., floor(log2(.5 * tfp.base() / ht.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phiT, Process::kf>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::tfp> tfp(setup);
    const Format<Variable::phiT, Process::ht> ht(setup);
    range_ = ht.range() + 2. * ht.base();
    base_ = ht.base() * pow(2., floor(log2(tfp.base() / ht.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::cot, Process::kf>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::cot, Process::tfp> tfp(setup);
    const Format<Variable::zT, Process::gp> zT(setup);
    const Format<Variable::r, Process::dtc> r(setup);
    range_ = (zT.base() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    base_ = zT.base() / r.base() * pow(2., floor(log2(tfp.base() / zT.base() * r.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::zT, Process::kf>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::zT, Process::tfp> tfp(setup);
    const Format<Variable::zT, Process::gp> gp(setup);
    range_ = gp.range();
    base_ = gp.base() * pow(2., floor(log2(tfp.base() / gp.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phi, Process::kf>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::phi, Process::dtc> phi(setup);
    const Format<Variable::phiT, Process::kf> phiT(setup);
    const Format<Variable::inv2R, Process::kf> inv2R(setup);
    range_ = phiT.base() + setup->maxRphi() * inv2R.base();
    base_ = phi.base();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::match, Process::kf>::Format(const Setup* setup) : DataFormat(false) {
    width_ = 1;
  }

  template <>
  Format<Variable::cot, Process::dr>::Format(const Setup* setup) : DataFormat(true) {
    const Format<Variable::cot, Process::kf> kf(setup);
    const Format<Variable::zT, Process::kf> zT(setup);
    range_ = (zT.range() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    base_ = kf.base();
    width_ = ceil(log2(range_ / base_));
  }

}  // namespace trackerTFP
