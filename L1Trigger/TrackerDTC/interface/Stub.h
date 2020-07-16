#ifndef L1Trigger_TrackerDTC_Stub_h
#define L1Trigger_TrackerDTC_Stub_h

#include "L1Trigger/TrackerDTC/interface/Setup.h"

#include <utility>
#include <vector>

namespace trackerDTC {

  // representation of a stub
  class Stub {
  public:
    Stub(const edm::ParameterSet&, const Setup&, SensorModule*, const TTStubRef&);
    ~Stub() {}

    // underlying TTStubRef
    TTStubRef ttStubRef() const { return ttStubRef_; }
    // did pass pt and eta cut
    bool valid() const { return valid_; }
    // stub bend in quarter pitch units
    int bend() const { return bend_; }
    // bit accurate representation of Stub
    TTDTC::BV frame(int region) const;
    // checks stubs region assignment
    bool inRegion(int region) const;

  private:
    // truncates double precision to f/w integer equivalent
    double digi(double value, double precision) const;
    // 64 bit stub in hybrid data format
    TTDTC::BV formatHybrid(int region) const;
    // 64 bit stub in tmtt data format
    TTDTC::BV formatTMTT(int region) const;

    // stores, calculates and provides run-time constants
    const Setup* setup_;
    // representation of an outer tracker sensormodule
    SensorModule* sm_;
    // underlying TTStubRef
    TTStubRef ttStubRef_;
    // chosen TT algorithm
    bool hybrid_;
    // passes pt and eta cut
    bool valid_;
    // column number in pitch units
    int col_;
    // row number in half pitch units
    int row_;
    // bend number in quarter pitch units
    int bend_;
    // reduced row number for look up
    int rowLUT_;
    // sub row number inside reduced row number
    int rowSub_;
    // stub r w.r.t. an offset in cm
    double r_;
    // stub phi w.r.t. detector region centre in rad
    double phi_;
    // stub z w.r.t. an offset in cm
    double z_;
    // slope of linearized stub phi in rad / pitch
    double m_;
    // intercept of linearized stub phi in rad
    double c_;
    // radius of a column of strips/pixel in cm
    double d_;
    // range of stub qOverPt in 1/cm
    std::pair<double, double> qOverPt_;
    // range of stub cot(theta)
    std::pair<double, double> cot_;
    // range of stub extrapolated phi to radius chosenRofPhi in rad
    std::pair<double, double> phiT_;
    // shared regions this stub belongs to [0-1]
    std::vector<int> regions_;
  };

}  // namespace trackerDTC

#endif