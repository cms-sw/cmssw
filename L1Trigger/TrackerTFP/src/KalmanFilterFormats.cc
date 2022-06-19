#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <limits>
#include <cstring>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  constexpr auto variableKFstrs_ = {
      "x0",        "x1",     "x2",     "x3",  "H00",      "H12",      "m0",           "m1",           "v0",
      "v1",        "r0",     "r1",     "S00", "S01",      "S12",      "S13",          "K00",          "K10",
      "K21",       "K31",    "R00",    "R11", "R00Rough", "R11Rough", "invR00Approx", "invR11Approx", "invR00Cor",
      "invR11Cor", "invR00", "invR11", "C00", "C01",      "C11",      "C22",          "C23",          "C33"};

  void KalmanFilterFormats::endJob() {
    const int wName =
        strlen(*max_element(variableKFstrs_.begin(), variableKFstrs_.end(), [](const auto& a, const auto& b) {
          return strlen(a) < strlen(b);
        }));
    static constexpr int wWidth = 3;
    for (VariableKF v = VariableKF::begin; v != VariableKF::end; v = VariableKF(+v + 1)) {
      const pair<double, double>& range = format(v).rangeActual();
      const double r = format(v).twos() ? max(abs(range.first), abs(range.second)) * 2. : range.second;
      const int width = ceil(log2(r / format(v).base()));
      cout << setw(wName) << *next(variableKFstrs_.begin(), +v) << ": " << setw(wWidth) << width << " " << setw(wWidth)
           << format(v).width() << " | " << setw(wWidth) << format(v).width() - width << endl;
    }
  }

  KalmanFilterFormats::KalmanFilterFormats() : iConfig_(), dataFormats_(nullptr), setup_(nullptr) {
    formats_.reserve(+VariableKF::end);
  }

  KalmanFilterFormats::KalmanFilterFormats(const ParameterSet& iConfig, const DataFormats* dataFormats)
      : iConfig_(dataFormats->hybrid() ? iConfig.getParameter<ParameterSet>("hybrid")
                                       : iConfig.getParameter<ParameterSet>("tmtt")),
        dataFormats_(dataFormats),
        setup_(dataFormats_->setup()) {
    formats_.reserve(+VariableKF::end);
    fillFormats();
  }

  template <VariableKF it>
  void KalmanFilterFormats::fillFormats() {
    formats_.emplace_back(FormatKF<it>(dataFormats_, iConfig_));
    if constexpr (++it != VariableKF::end)
      fillFormats<++it>();
  }

  DataFormatKF::DataFormatKF(const VariableKF& v, bool twos)
      : v_(v),
        twos_(twos),
        width_(0),
        base_(1.),
        range_(0.),
        rangeActual_(numeric_limits<double>::max(), numeric_limits<double>::lowest()) {}

  // returns false if data format would oferflow for this double value
  bool DataFormatKF::inRange(double d) const {
    if (twos_)
      return d >= -range_ / 2. && d < range_ / 2.;
    return d >= 0 && d < range_;
  }

  void DataFormatKF::updateRangeActual(double d) {
    rangeActual_ = make_pair(min(rangeActual_.first, d), max(rangeActual_.second, d));
    if (!inRange(d)) {
      string v = *next(variableKFstrs_.begin(), +v_);
      cms::Exception exception("out_of_range");
      exception.addContext("trackerTFP:DataFormatKF::updateRangeActual");
      exception << "Variable " << v << " = " << d << " is out of range " << (twos_ ? -range_ / 2. : 0) << " to "
                << (twos_ ? range_ / 2. : range_) << "." << endl;
      if (twos_ || d >= 0.)
        exception.addAdditionalInfo("Consider raising BaseShift" + v + " in KalmnaFilterFormats_cfi.py.");
      throw exception;
    }
  }

  template <>
  FormatKF<VariableKF::x0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x0, true) {
    const DataFormat& input = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx0");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x1, true) {
    const DataFormat& input = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx1");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x2, true) {
    const DataFormat& input = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx2");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::x3>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::x3, true) {
    const DataFormat& input = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx3");
    base_ = pow(2, baseShift) * input.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::H00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::H00, true) {
    const DataFormat& kfin = dataFormats->format(Variable::r, Process::kfin);
    base_ = kfin.base();
    width_ = kfin.width();
    range_ = kfin.range();
  }

  template <>
  FormatKF<VariableKF::H12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::H12, true) {
    const Setup* setup = dataFormats->setup();
    const DataFormat& kfin = dataFormats->format(Variable::r, Process::kfin);
    base_ = kfin.base();
    range_ = 2. * max(abs(setup->outerRadius() - setup->chosenRofZ()), abs(setup->innerRadius() - setup->chosenRofZ()));
    width_ = ceil(log2(range_ / base_));
  }

  template <>
  FormatKF<VariableKF::m0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::m0, true) {
    const DataFormat& kfin = dataFormats->format(Variable::phi, Process::kfin);
    base_ = kfin.base();
    width_ = kfin.width();
    range_ = kfin.range();
  }

  template <>
  FormatKF<VariableKF::m1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::m1, true) {
    const DataFormat& kfin = dataFormats->format(Variable::z, Process::kfin);
    base_ = kfin.base();
    width_ = kfin.width();
    range_ = kfin.range();
  }

  template <>
  FormatKF<VariableKF::v0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::v0, false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftv0");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::v1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::v1, true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftv1");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r0, true) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr0");
    base_ = pow(2., baseShift) * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::r1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::r1, true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr1");
    base_ = pow(2., baseShift) * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S00, true) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS00");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S01, true) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS01");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S12, true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS12");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::S13>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::S13, true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS13");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K00, true) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK00");
    base_ = pow(2., baseShift) * x0.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K10>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K10, true) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK10");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K21, true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK21");
    base_ = pow(2., baseShift) * x2.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::K31>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::K31, true) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK31");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPab();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R00, false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR00");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R11, false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR11");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::R00Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R00Rough, false) {
    const FormatKF<VariableKF::R00> R00(dataFormats, iConfig);
    width_ = dataFormats->setup()->widthAddrBRAM18();
    range_ = R00.range();
    const int baseShift = R00.width() - width_;
    base_ = pow(2., baseShift) * R00.base();
  }

  template <>
  FormatKF<VariableKF::R11Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::R11Rough, false) {
    const FormatKF<VariableKF::R11> R11(dataFormats, iConfig);
    width_ = dataFormats->setup()->widthAddrBRAM18();
    range_ = R11.range();
    const int baseShift = R11.width() - width_;
    base_ = pow(2., baseShift) * R11.base();
  }

  template <>
  FormatKF<VariableKF::invR00Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00Approx, false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00Approx");
    base_ = pow(2., baseShift) / x1.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11Approx, false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11Approx");
    base_ = pow(2., baseShift) / x3.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR00Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00Cor, false) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00Cor");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11Cor, false) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11Cor");
    base_ = pow(2., baseShift);
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR00, false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00");
    base_ = pow(2., baseShift) / x1.base() / x1.base();
    width_ = dataFormats->setup()->widthDSPau();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::invR11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::invR11, false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11");
    base_ = pow(2., baseShift) / x3.base() / x3.base();
    width_ = dataFormats->setup()->widthDSPau();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C00, false) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC00");
    base_ = pow(2., baseShift) * x0.base() * x0.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C01, true) {
    const DataFormat& x0 = dataFormats->format(Variable::inv2R, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC01");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C11, false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC11");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C22>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C22, false) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC22");
    base_ = pow(2., baseShift) * x2.base() * x2.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C23>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C23, true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC23");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbb();
    calcRange();
  }

  template <>
  FormatKF<VariableKF::C33>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig)
      : DataFormatKF(VariableKF::C33, false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC33");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
    width_ = dataFormats->setup()->widthDSPbu();
    calcRange();
  }

}  // namespace trackerTFP
