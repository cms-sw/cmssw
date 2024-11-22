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

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  // default constructor, trying to need space as proper constructed object
  DataFormats::DataFormats()
      : numDataFormats_(0),
        formats_(+Variable::end, std::vector<DataFormat*>(+Process::end, nullptr)),
        numUnusedBitsStubs_(+Process::end, TTBV::S_ - 1),
        numUnusedBitsTracks_(+Process::end, TTBV::S_ - 1) {
    channelAssignment_ = nullptr;
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
  DataFormats::DataFormats(const ChannelAssignment* channelAssignment) : DataFormats() {
    channelAssignment_ = channelAssignment;
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
      dataFormats_.emplace_back(Format<v, p>(channelAssignment_));
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
  Format<Variable::inv2R, Process::tfp>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kRinvSize;
    range_ = -2. * TTTrack_TrackWord::minRinv;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::phiT, Process::tfp>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kPhiSize;
    range_ = -2. * TTTrack_TrackWord::minPhi0;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::cot, Process::tfp>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kTanlSize;
    range_ = -2. * TTTrack_TrackWord::minTanl;
    base_ = range_ * pow(2, -width_);
  }
  template <>
  Format<Variable::zT, Process::tfp>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    width_ = TTTrack_TrackWord::TrackBitWidths::kZ0Size;
    range_ = -2. * TTTrack_TrackWord::minZ0;
    base_ = range_ * pow(2, -width_);
  }

  template <>
  Format<Variable::inv2R, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    static const double thight = s->htNumBinsInv2R();
    static const double loose = thight + 2;
    range_ = 2. * s->invPtToDphi() / s->minPt() * loose / thight;
    base_ = range_ / loose;
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phiT, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    static const double thight = s->gpNumBinsPhiT() * s->htNumBinsPhiT();
    static const double loose = thight + 2 * s->gpNumBinsPhiT();
    range_ = 2. * M_PI / s->numRegions() * loose / thight;
    base_ = range_ / loose;
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::zT, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    static const double thight = s->gpNumBinsZT();
    static const double loose = thight + 2;
    range_ = 2. * sinh(s->maxEta()) * s->chosenRofZ() * loose / thight;
    base_ = range_ / loose;
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::cot, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    const Format<Variable::zT, Process::tm> zT(ca);
    range_ = (zT.range() + 2. * s->beamWindowZ()) / s->chosenRofZ();
    base_ = (zT.base() + 2. * s->beamWindowZ()) / s->chosenRofZ();
  }

  template <>
  Format<Variable::stubId, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(false) {
    width_ = ca->tmWidthStubId() + 1;
  }
  template <>
  Format<Variable::r, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    const Format<Variable::phiT, Process::tm> phiT(ca);
    const Format<Variable::inv2R, Process::tm> inv2R(ca);
    range_ = 2. * s->maxRphi();
    base_ = phiT.base() / inv2R.base();
    const int shift = ceil(log2(range_ / base_)) - s->tmttWidthR();
    base_ *= pow(2., shift);
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phi, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    const Format<Variable::phiT, Process::tm> phiT(ca);
    const Format<Variable::inv2R, Process::tm> inv2R(ca);
    const double range = s->baseRegion() + s->maxRphi() * inv2R.range();
    range_ = phiT.base() + s->maxRphi() * inv2R.base();
    const int shift = ceil(log2(range / phiT.base())) - s->tmttWidthPhi();
    base_ = phiT.base() * pow(2., shift);
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::z, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Setup* s = ca->setup();
    const Format<Variable::zT, Process::tm> zT(ca);
    const Format<Variable::cot, Process::tm> cot(ca);
    const double range = 2. * s->halfLength();
    range_ = zT.base() + s->maxRz() * cot.base();
    const int shift = ceil(log2(range / zT.base())) - s->tmttWidthZ();
    base_ = zT.base() * pow(2., shift);
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::dPhi, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(false) {
    const Setup* s = ca->setup();
    const Format<Variable::phi, Process::tm> phi(ca);
    const Format<Variable::inv2R, Process::tm> inv2R(ca);
    range_ = .5 * s->pitchRowPS() / s->innerRadius() + .25 * (s->pitchCol2S() + s->scattering()) * inv2R.range();
    base_ = phi.base();
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::dZ, Process::tm>::Format(const ChannelAssignment* ca) : DataFormat(false) {
    const Setup* s = ca->setup();
    const Format<Variable::z, Process::tm> z(ca);
    range_ = .5 * s->pitchCol2S() * sinh(s->maxEta());
    base_ = z.base();
    width_ = ceil(log2(range_ / base_));
  }

  template <>
  Format<Variable::inv2R, Process::kf>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Format<Variable::inv2R, Process::tfp> tfp(ca);
    const Format<Variable::inv2R, Process::tm> tm(ca);
    range_ = tm.range();
    base_ = tm.base() * pow(2., floor(log2(.5 * tfp.base() / tm.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::phiT, Process::kf>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Format<Variable::phiT, Process::tfp> tfp(ca);
    const Format<Variable::phiT, Process::tm> tm(ca);
    range_ = tm.range();
    base_ = tm.base() * pow(2., floor(log2(tfp.base() / tm.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::cot, Process::kf>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Format<Variable::cot, Process::tfp> tfp(ca);
    const Format<Variable::cot, Process::tm> cot(ca);
    const Format<Variable::z, Process::tm> z(ca);
    const Format<Variable::r, Process::tm> r(ca);
    range_ = cot.range();
    base_ = z.base() / r.base() * pow(2., floor(log2(tfp.base() / z.base() * r.base())));
    width_ = ceil(log2(range_ / base_));
  }
  template <>
  Format<Variable::zT, Process::kf>::Format(const ChannelAssignment* ca) : DataFormat(true) {
    const Format<Variable::zT, Process::tfp> tfp(ca);
    const Format<Variable::zT, Process::tm> tm(ca);
    range_ = tm.range();
    base_ = tm.base() * pow(2., floor(log2(tfp.base() / tm.base())));
    width_ = ceil(log2(range_ / base_));
  }

}  // namespace trklet
