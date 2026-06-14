#ifndef L1Trigger_TrackFindingTracklet_DataFormats_h
#define L1Trigger_TrackFindingTracklet_DataFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Hybrid emulator
enabling automated conversions from frames to stubs/tracks and vice versa
In data members of classes Stub* & Track* below, the variables describing
stubs/tracks are stored both in digitial format as a 64b word in frame_,
and in undigitized format in an std::tuple. (This saves CPU)
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <vector>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <iostream>
#include <string>

namespace trklet {

  // hybrid processes
  enum class Process { begin, dr = begin, kf, tq, tfp, end, x };
  // hybrid variables
  enum class Variable {
    begin,
    r = begin,
    phi,
    z,
    dPhi,
    dZ,
    layerId,
    seedType,
    inv2R,
    phi0,
    cot,
    z0,
    chi20,
    chi21,
    mva,
    hitPattern,
    end,
    x
  };
  // hybrid process order
  constexpr std::initializer_list<Process> Processes = {Process::dr, Process::kf, Process::tq, Process::tfp};
  // conversion: Process to int
  inline constexpr int operator+(Process p) { return static_cast<int>(p); }
  // conversion: Variable to int
  inline constexpr int operator+(Variable v) { return static_cast<int>(v); }
  // increment of Process
  inline constexpr Process operator++(Process p) { return Process(+p + 1); }
  // increment of Variable
  inline constexpr Variable operator++(Variable v) { return Variable(+v + 1); }

  //Base class representing format of a variable
  class DataFormat {
  public:
    DataFormat(bool twos, int width)
        : twos_(twos), width_(width), base_(1.), range_(std::pow(2, width)), coverage_(range_) {}
    DataFormat(bool twos, int width, double base, double range)
        : twos_(twos), width_(width), base_(base), range_(range), coverage_(base * std::pow(2, width)) {}
    DataFormat(bool twos, int width, double range)
        : twos_(twos), width_(width), base_(range * std::pow(2, -width)), range_(range), coverage_(range_) {}
    DataFormat(bool twos, double base, double range)
        : twos_(twos),
          width_(std::ceil(std::log2(range / base))),
          base_(base),
          range_(range),
          coverage_(base * std::pow(2, width_)) {}
    DataFormat() {}
    ~DataFormat() = default;
    // converts int to bitvector
    TTBV ttBV(int i) const { return TTBV(i, width_, twos_); }
    // converts double to bitvector
    TTBV ttBV(double d) const { return TTBV(d, base_, width_, twos_); }
    // extracts int from bitvector, removing these bits from bitvector
    void extract(TTBV& in, int& out) const { out = in.extract(width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, double& out) const { out = in.extract(base_, width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, TTBV& out) const { out = in.slice(width_, twos_); }
    // extracts bool from bitvector, removing these bits from bitvector
    void extract(TTBV& in, bool& out) const { out = in.extract(); }
    // attaches integer to bitvector
    void attach(const int i, TTBV& ttBV) const { ttBV += TTBV(i, width_, twos_); }
    // attaches double to bitvector
    void attach(const double d, TTBV& ttBV) const { ttBV += TTBV(d, base_, width_, twos_); }
    // attaches bitvector to bitvector
    void attach(const TTBV& bv, TTBV& ttBV) const { ttBV += bv; }
    // converts int to double
    double floating(int i) const { return (i + .5) * base_; }
    // converts double to int
    int integer(double d) const { return tt::floor(d / base_); }
    // converts double to int and back to double
    double digi(double d) const { return floating(integer(d)); }
    // converts binary integer value to twos complement integer value
    int toSigned(int i) const { return i - std::pow(2, width_) / 2; }
    // converts twos complement integer value to binary integer value
    int toUnsigned(int i) const { return i + std::pow(2, width_) / 2; }
    // converts floating point value to binary integer value
    int toUnsigned(double d) const { return this->integer(d) + std::pow(2, width_) / 2; }
    // limit to biggest representable floating point value
    double limit(double d) const {
      if (this->isCovered(d))
        return d;
      if (twos_) {
        if (d < 0.)
          return (base_ - range_) / 2.;
        return (range_ - base_) / 2.;
      }
      return range_ - base_ / 2.;
    }
    // scales value from 0 to 1 (unsigned) or from -1 to 1 (signed)
    double scale(double d) const { return d / coverage_; }
    // scales absolute values from 0 to 1
    double scaleABS(double d) const {
      if (twos_)
        return 2. * std::abs(d) / coverage_;
      return this->scale(d);
    }
    // returns false if data outside expected range
    bool inRange(double d) const {
      if (twos_)
        return d >= -range_ / 2. && d < range_ / 2.;
      return d < range_;
    }
    // returns false if data outside representable range
    bool isCovered(double d) const {
      if (twos_)
        return d >= -coverage_ / 2. && d < coverage_ / 2.;
      return d < coverage_;
    }
    // true if twos'complement or false if binary representation is chosen
    bool twos() const { return twos_; }
    // number of used bits
    int width() const { return width_; }
    // precision
    double base() const { return base_; }
    // needed range
    double range() const { return range_; }
    // covered range
    double coverage() const { return coverage_; }

  protected:
    // true if twos'complement or false if binary representation is chosen
    bool twos_;
    // number of used bits
    int width_;
    // precision
    double base_;
    // needed range
    double range_;
    // covered range
    double coverage_;
  };

  // function template for DataFormat generation
  template <Variable v, Process p>
  DataFormat makeDataFormat(const Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::tfp>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi0, Process::tfp>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::tfp>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z0, Process::tfp>(const Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::r, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::dPhi, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::dZ, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::seedType, Process::dr>(const Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::inv2R, Process::kf>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi0, Process::kf>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::cot, Process::kf>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z0, Process::kf>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::phi, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::z, Process::dr>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::layerId, Process::kf>(const Setup* setup);

  template <>
  DataFormat makeDataFormat<Variable::chi20, Process::tq>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::chi21, Process::tq>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::mva, Process::tq>(const Setup* setup);
  template <>
  DataFormat makeDataFormat<Variable::hitPattern, Process::tq>(const Setup* setup);

  /*! \class  trklet::DataFormats
   *  \brief  Class to calculate and provide dataformats used by Hybrid emulator
   *  \author Thomas Schuh
   *  \date   2024, Sep
   */
  class DataFormats {
  private:
    // variable flavour mapping, Each row below declares which processing steps use the variable named in the comment at the end of the row
    static constexpr std::array<std::array<Process, +Process::end>, +Variable::end> config_ = {{
        // Process::dr Process::kf Process::tq Process::tfp
        {{Process::dr, Process::dr, Process::dr, Process::x}},   // Variable::r
        {{Process::dr, Process::kf, Process::kf, Process::x}},   // Variable::phi
        {{Process::dr, Process::kf, Process::kf, Process::x}},   // Variable::z
        {{Process::dr, Process::dr, Process::dr, Process::x}},   // Variable::dPhi
        {{Process::dr, Process::dr, Process::dr, Process::x}},   // Variable::dZ
        {{Process::x, Process::kf, Process::x, Process::x}},     // Variable::layerId
        {{Process::dr, Process::x, Process::x, Process::x}},     // Variable::seedType
        {{Process::x, Process::kf, Process::kf, Process::tfp}},  // Variable::inv2R
        {{Process::x, Process::kf, Process::kf, Process::tfp}},  // Variable::phi0
        {{Process::x, Process::kf, Process::kf, Process::tfp}},  // Variable::cot
        {{Process::x, Process::kf, Process::kf, Process::tfp}},  // Variable::z0
        {{Process::x, Process::x, Process::tq, Process::x}},     // Variable::chi20
        {{Process::x, Process::x, Process::tq, Process::x}},     // Variable::chi21
        {{Process::x, Process::x, Process::tq, Process::x}},     // Variable::mva
        {{Process::x, Process::x, Process::tq, Process::x}}      // Variable::hitPattern
    }};
    // stub word assembly, shows which stub variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> stubs_ = {{
        {Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},                     // Process::dr
        {Variable::layerId, Variable::r, Variable::phi, Variable::z, Variable::dPhi, Variable::dZ},  // Process::kf
        {},                                                                                          // Process::tq
        {}                                                                                           // Process::tfp
    }};
    // track word assembly, shows which track variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> tracks_ = {{
        {Variable::seedType},                                                     // Process::dr
        {Variable::inv2R, Variable::phi0, Variable::cot, Variable::z0},           // Process::kf
        {Variable::hitPattern, Variable::mva, Variable::chi20, Variable::chi21, Variable::z0, Variable::cot},  // Process::tq
        {}                                                                        // Process::tfp
    }};

  public:
    DataFormats();
    DataFormats(const Setup* setup);
    ~DataFormats() = default;
    // converts bits to ntuple of variables
    template <typename... Ts>
    void convertStub(Process p, const tt::Frame& bv, std::tuple<Ts...>& data) const {
      TTBV ttBV(bv);
      extractStub(p, ttBV, data);
    }
    // converts ntuple of variables to bits
    template <typename... Ts>
    void convertStub(Process p, const std::tuple<Ts...>& data, tt::Frame& bv) const {
      TTBV ttBV(1, 1 + numUnusedBitsStubs_[+p]);
      attachStub(p, data, ttBV);
      bv = ttBV.bs();
    }
    // converts bits to ntuple of variables
    template <typename... Ts>
    void convertTrack(Process p, const tt::Frame& bv, std::tuple<Ts...>& data) const {
      TTBV ttBV(bv);
      extractTrack(p, ttBV, data);
    }
    // converts ntuple of variables to bits
    template <typename... Ts>
    void convertTrack(Process p, const std::tuple<Ts...>& data, tt::Frame& bv) const {
      TTBV ttBV(1, 1 + numUnusedBitsTracks_[+p]);
      attachTrack(p, data, ttBV);
      bv = ttBV.bs();
    }
    // access to run-time constants
    const Setup* setup() const { return setup_; }
    // number of bits being used for specific variable flavour
    int width(Variable v, Process p) const { return formats_[+v][+p]->width(); }
    // precision being used for specific variable flavour
    double base(Variable v, Process p) const { return formats_[+v][+p]->base(); }
    // covered range for specific variable flavour
    double range(Variable v, Process p) const { return formats_[+v][+p]->range(); }
    // access to spedific format
    const DataFormat& format(Variable v, Process p) const { return *formats_[+v][+p]; }

  private:
    // number of unique data formats
    int numDataFormats_;
    // method to count number of unique data formats
    template <Variable v = Variable::begin, Process p = Process::begin>
    void countFormats();
    // constructs data formats of all unique used variables and flavours
    template <Variable v = Variable::begin, Process p = Process::begin>
    void fillDataFormats();
    // helper (loop) data formats of all unique used variables and flavours
    template <Variable v, Process p, Process it = Process::begin>
    void fillFormats();
    // helper (loop) to convert bits to ntuple of variables
    template <int it = 0, typename... Ts>
    void extractStub(Process p, TTBV& ttBV, std::tuple<Ts...>& data) const {
      Variable v = *std::next(stubs_[+p].begin(), sizeof...(Ts) - 1 - it);
      formats_[+v][+p]->extract(ttBV, std::get<sizeof...(Ts) - 1 - it>(data));
      if constexpr (it + 1 != sizeof...(Ts))
        extractStub<it + 1>(p, ttBV, data);
    }
    // helper (loop) to convert bits to ntuple of variables
    template <int it = 0, typename... Ts>
    void extractTrack(Process p, TTBV& ttBV, std::tuple<Ts...>& data) const {
      Variable v = *std::next(tracks_[+p].begin(), sizeof...(Ts) - 1 - it);
      formats_[+v][+p]->extract(ttBV, std::get<sizeof...(Ts) - 1 - it>(data));
      if constexpr (it + 1 != sizeof...(Ts))
        extractTrack<it + 1>(p, ttBV, data);
    }
    // helper (loop) to convert ntuple of variables to bits
    template <int it = 0, typename... Ts>
    void attachStub(Process p, const std::tuple<Ts...>& data, TTBV& ttBV) const {
      Variable v = *std::next(stubs_[+p].begin(), it);
      formats_[+v][+p]->attach(std::get<it>(data), ttBV);
      if constexpr (it + 1 != sizeof...(Ts))
        attachStub<it + 1>(p, data, ttBV);
    }
    // helper (loop) to convert ntuple of variables to bits
    template <int it = 0, typename... Ts>
    void attachTrack(Process p, const std::tuple<Ts...>& data, TTBV& ttBV) const {
      Variable v = *std::next(tracks_[+p].begin(), it);
      formats_[+v][+p]->attach(std::get<it>(data), ttBV);
      if constexpr (it + 1 != sizeof...(Ts))
        attachTrack<it + 1>(p, data, ttBV);
    }
    // stored run-time constants
    const Setup* setup_;
    // collection of unique formats
    std::vector<DataFormat> dataFormats_;
    // variable flavour mapping
    std::vector<std::vector<DataFormat*>> formats_;
    // number of unused frame bits for a all Stub flavours
    std::vector<int> numUnusedBitsStubs_;
    // number of unused frame bits for a all Track flavours
    std::vector<int> numUnusedBitsTracks_;
  };

  // base class to represent stubs
  template <typename... Ts>
  class BaseStub {
  public:
    // construct Stub from Frame
    BaseStub(const tt::FrameStub& fs, const DataFormats* df, Process p) : dataFormats_(df), p_(p), frame_(fs) {
      dataFormats_->convertStub(p_, frame_.second, data_);
    }
    template <typename... Others>
    // construct Stub from other Stub
    BaseStub(const BaseStub<Others...>& stub, Ts... data)
        : dataFormats_(stub.dataFormats()), p_(++stub.p()), frame_(stub.frame()), data_(data...) {
      dataFormats_->convertStub(p_, data_, frame_.second);
    }
    // construct Stub from TTStubRef
    BaseStub(const TTStubRef& ttStubRef, const DataFormats* df, Process p, Ts... data)
        : dataFormats_(df), p_(p), frame_(ttStubRef, tt::Frame()), data_(data...) {
      dataFormats_->convertStub(p_, data_, frame_.second);
    }
    BaseStub() {}
    virtual ~BaseStub() = default;
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // stub flavour
    Process p() const { return p_; }
    // acess to frame
    const tt::FrameStub& frame() const { return frame_; }

  protected:
    // all dataformats
    const DataFormats* dataFormats_;
    // stub flavour
    Process p_;
    // underlying TTStubRef and bitvector
    tt::FrameStub frame_;
    // ntuple of variables this stub is assemled of
    std::tuple<Ts...> data_;
  };

  // class to represent stubs generated by process DuplicateRemoval
  class StubDR : public BaseStub<double, double, double, double, double> {
  public:
    // construct StubDR from Frame
    StubDR(const tt::FrameStub& fs, const DataFormats* df) : BaseStub(fs, df, Process::dr) {}
    // construct StubTM from TTStubRef
    StubDR(const TTStubRef& ttStubRef, const DataFormats* df, double r, double phi, double z, double dPhi, double dZ)
        : BaseStub(ttStubRef, df, Process::dr, r, phi, z, dPhi, dZ) {}
    ~StubDR() override = default;
    // stub radius in cm wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual in rad
    double phi() const { return std::get<1>(data_); }
    // stub z residual in cm
    double z() const { return std::get<2>(data_); }
    // stub phi uncertainty in rad
    double dPhi() const { return std::get<3>(data_); }
    // stub z uncertainty in cm
    double dZ() const { return std::get<4>(data_); }
  };

  // class to represent stubs generated by process KalmanFilter
  class StubKF : public BaseStub<int, double, double, double, double, double> {
  public:
    // construct StubKF from Frame
    StubKF(const tt::FrameStub& fs, const DataFormats* df) : BaseStub(fs, df, Process::kf) {}
    // construct StubKF from StubDR
    StubKF(const StubDR& stub, int layerId, double r, double phi, double z, double dPhi, double dZ)
        : BaseStub(stub, layerId, r, phi, z, dPhi, dZ) {}
    ~StubKF() override = default;
    // reduced 7 bit layer id [0 = {1}, 1 = {2}, 2 = {11 or 6}, 3 = {12 or 5}, 4 = {13 or 4}, 5 = {14}, 6 = {15 or 3}]
    int layerId() const { return std::get<0>(data_); };
    // stub radius in cm wrt chosenRofPhi
    double r() const { return std::get<1>(data_); };
    // stub phi residual in rad
    double phi() const { return std::get<2>(data_); };
    // stub z residual in cm
    double z() const { return std::get<3>(data_); };
    // stub phi uncertainty in rad
    double dPhi() const { return std::get<4>(data_); }
    // stub z uncertainty in cm
    double dZ() const { return std::get<5>(data_); }
  };

  // base class to represent tracks
  template <typename... Ts>
  class BaseTrack {
  public:
    // construct Track from Frame
    BaseTrack(const tt::FrameTrack& ft, const DataFormats* df, Process p) : dataFormats_(df), p_(p), frame_(ft) {
      dataFormats_->convertTrack(p_, frame_.second, data_);
    }
    // construct Track from TTTrackRef
    BaseTrack(const TTTrackRef& ttTrackRef, const DataFormats* df, Process p, Ts... data)
        : dataFormats_(df), p_(p), frame_(ttTrackRef, tt::Frame()), data_(data...) {
      dataFormats_->convertTrack(p_, data_, frame_.second);
    }
    // construct Track from other Track
    template <typename... Others>
    BaseTrack(const BaseTrack<Others...>& track, Ts... data)
        : dataFormats_(track.dataFormats()), p_(++track.p()), frame_(track.frame()), data_(data...) {
      dataFormats_->convertTrack(p_, data_, frame_.second);
    }
    BaseTrack() {}
    virtual ~BaseTrack() = default;
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // track flavour
    Process p() const { return p_; }
    // acces to frame
    const tt::FrameTrack& frame() const { return frame_; }

  protected:
    // all data formats
    const DataFormats* dataFormats_;
    // track flavour
    Process p_;
    // underlying TTTrackRef and bitvector
    tt::FrameTrack frame_;
    // ntuple of variables this track is assemled of
    std::tuple<Ts...> data_;
  };

  // class to represent tracks generated by process DuplicateRemoval
  class TrackDR : public BaseTrack<int> {
  public:
    // construct TrackKF from Frame
    TrackDR(const tt::FrameTrack& ft, const DataFormats* df) : BaseTrack(ft, df, Process::dr) {}
    // construct TrackKF from TTTrackRef
    TrackDR(const TTTrackRef& ttTrackRef, const DataFormats* df, int seedType)
        : BaseTrack(ttTrackRef, df, Process::dr, seedType) {}
    TrackDR() {}
    ~TrackDR() override = default;
    // seed type
    int seedType() const { return std::get<0>(data_); }
  };

  // class to represent tracks generated by process KalmanFilter
  class TrackKF : public BaseTrack<double, double, double, double> {
  public:
    // construct TrackKF from Frame
    TrackKF(const tt::FrameTrack& ft, const DataFormats* df) : BaseTrack(ft, df, Process::kf) {}
    // construct TrackKF from TrackDR
    TrackKF(const TrackDR& track, double inv2R, double phiT, double cot, double zT)
        : BaseTrack(track, inv2R, phiT, cot, zT) {}
    TrackKF() {}
    ~TrackKF() override = default;
    // track inv2R in 1/cm
    double inv2R() const { return std::get<0>(data_); }
    // track phi wrt pprocessing centre in rad
    double phi0() const { return std::get<1>(data_); }
    // track cotThea
    double cot() const { return std::get<2>(data_); }
    // track z in cm
    double z0() const { return std::get<3>(data_); }
  };

  // class to represent tracks generated by process TrackQuality
  class TrackTQ : public BaseTrack<TTBV, int, double, double, double, double> {
  public:
    // construct TrackTQ from Frame
    TrackTQ(const tt::FrameTrack& ft, const DataFormats* df) : BaseTrack(ft, df, Process::tq) {}
    // construct TrackTQ from TrackKF
    TrackTQ(const TrackKF& track, const TTBV& hitPattern, int mva, double chi20, double chi21, double z0, double cot)
        : BaseTrack(track, hitPattern, mva, chi20, chi21, z0, cot) {}
    TrackTQ() {}
    ~TrackTQ() override = default;
    // mva
    const TTBV& hitPattern() const { return std::get<0>(data_); }
    // mva
    int mva() const { return std::get<1>(data_); }
    // track r-phi chi2
    double chi20() const { return std::get<2>(data_); }
    // track r-z chi2
    double chi21() const { return std::get<3>(data_); }
    // 
    double z0() const { return std::get<4>(data_); }
    //
    double cot() const { return std::get<5>(data_); }
  };

}  // namespace trklet

EVENTSETUP_DATA_DEFAULT_RECORD(trklet::DataFormats, trackerDTC::SetupRcd);

#endif
