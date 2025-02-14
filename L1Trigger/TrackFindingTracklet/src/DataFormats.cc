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
      dataFormats_.emplace_back(makeDataFormat<v, p>(channelAssignment_));
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
  DataFormat makeDataFormat<Variable::inv2R, Process::tfp>(const ChannelAssignment* ca) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kRinvSize;
    const double range = -2. * TTTrack_TrackWord::minRinv;
    const double base = range * pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::tfp>(const ChannelAssignment* ca) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kPhiSize;
    const double range = -2. * TTTrack_TrackWord::minPhi0;
    const double base = range * pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tfp>(const ChannelAssignment* ca) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kTanlSize;
    const double range = -2. * TTTrack_TrackWord::minTanl;
    const double base = range * pow(2, -width);
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::tfp>(const ChannelAssignment* ca) {
    const int width = TTTrack_TrackWord::TrackBitWidths::kZ0Size;
    const double range = -2. * TTTrack_TrackWord::minZ0;
    const double base = range * pow(2, -width);
    return DataFormat(true, width, base, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const double thight = s->htNumBinsInv2R();
    const double loose = thight + 2;
    const double range = 2. * s->invPtToDphi() / s->minPt() * loose / thight;
    const double base = range / loose;
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const double thight = s->gpNumBinsPhiT() * s->htNumBinsPhiT();
    const double loose = thight + 2 * s->gpNumBinsPhiT();
    const double range = 2. * M_PI / s->numRegions() * loose / thight;
    const double base = range / loose;
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const double thight = s->gpNumBinsZT();
    const double loose = thight + 2;
    const double range = 2. * sinh(s->maxEta()) * s->chosenRofZ() * loose / thight;
    const double base = range / loose;
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat zT = makeDataFormat<Variable::zT, Process::tm>(ca);
    const double range = (zT.range() + 2. * s->beamWindowZ()) / s->chosenRofZ();
    const double base = (zT.base() + 2. * s->beamWindowZ()) / s->chosenRofZ();
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::stubId, Process::tm>(const ChannelAssignment* ca) {
    const int width = ca->tmWidthStubId() + 1;
    const double base = 1.;
    const double range = pow(2, width);
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::r, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::tm>(ca);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::tm>(ca);
    const double range = 2. * s->maxRphi();
    const double baseShifted = phiT.base() / inv2R.base();
    const int shift = ceil(log2(range / baseShifted)) - s->tmttWidthR();
    const double base = baseShifted * pow(2., shift);
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat phiT = makeDataFormat<Variable::phiT, Process::tm>(ca);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::tm>(ca);
    const double rangeMin = s->baseRegion() + s->maxRphi() * inv2R.range();
    const double range = phiT.base() + s->maxRphi() * inv2R.base();
    const int shift = ceil(log2(rangeMin / phiT.base())) - s->tmttWidthPhi();
    const double base = phiT.base() * pow(2., shift);
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::z, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat zT = makeDataFormat<Variable::zT, Process::tm>(ca);
    const DataFormat cot = makeDataFormat<Variable::cot, Process::tm>(ca);
    const double rangeMin = 2. * s->halfLength();
    const double range = zT.base() + s->maxRz() * cot.base();
    const int shift = ceil(log2(rangeMin / zT.base())) - s->tmttWidthZ();
    const double base = zT.base() * pow(2., shift);
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::dPhi, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat phi = makeDataFormat<Variable::phi, Process::tm>(ca);
    const DataFormat inv2R = makeDataFormat<Variable::inv2R, Process::tm>(ca);
    const double range =
        .5 * s->pitchRowPS() / s->innerRadius() + .25 * (s->pitchCol2S() + s->scattering()) * inv2R.range();
    const double base = phi.base();
    const int width = ceil(log2(range / base));
    return DataFormat(false, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::dZ, Process::tm>(const ChannelAssignment* ca) {
    const Setup* s = ca->setup();
    const DataFormat z = makeDataFormat<Variable::z, Process::tm>(ca);
    const double range = .5 * s->pitchCol2S() * sinh(s->maxEta());
    const double base = z.base();
    const int width = ceil(log2(range / base));
    return DataFormat(false, width, base, range);
  }

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::kf>(const ChannelAssignment* ca) {
    const DataFormat tfp = makeDataFormat<Variable::inv2R, Process::tfp>(ca);
    const DataFormat tm = makeDataFormat<Variable::inv2R, Process::tm>(ca);
    const double range = tm.range();
    const double base = tm.base() * pow(2., floor(log2(.5 * tfp.base() / tm.base())));
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::phiT, Process::kf>(const ChannelAssignment* ca) {
    const DataFormat tfp = makeDataFormat<Variable::phiT, Process::tfp>(ca);
    const DataFormat tm = makeDataFormat<Variable::phiT, Process::tm>(ca);
    const double range = tm.range();
    const double base = tm.base() * pow(2., floor(log2(tfp.base() / tm.base())));
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::kf>(const ChannelAssignment* ca) {
    const DataFormat tfp = makeDataFormat<Variable::cot, Process::tfp>(ca);
    const DataFormat cot = makeDataFormat<Variable::cot, Process::tm>(ca);
    const DataFormat z = makeDataFormat<Variable::z, Process::tm>(ca);
    const DataFormat r = makeDataFormat<Variable::r, Process::tm>(ca);
    const double range = cot.range();
    const double base = z.base() / r.base() * pow(2., floor(log2(tfp.base() / z.base() * r.base())));
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }
  template <>
  DataFormat makeDataFormat<Variable::zT, Process::kf>(const ChannelAssignment* ca) {
    const DataFormat tfp = makeDataFormat<Variable::zT, Process::tfp>(ca);
    const DataFormat tm = makeDataFormat<Variable::zT, Process::tm>(ca);
    const double range = tm.range();
    const double base = tm.base() * pow(2., floor(log2(tfp.base() / tm.base())));
    const int width = ceil(log2(range / base));
    return DataFormat(true, width, base, range);
  }

}  // namespace trklet
