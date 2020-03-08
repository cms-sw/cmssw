#ifndef __L1TTrackerDTC_STUB_H__
#define __L1TTrackerDTC_STUB_H__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <utility>
#include <vector>


namespace L1TTrackerDTC {

  class Settings;
  class Module;

  // representation of a stub
  class Stub {
  public:
    Stub(Settings* settings, const TTStubRef& ttStubRef, Module* module);

    ~Stub() {}

    // access to data members
    TTStubRef ttStubRef() const { return ttStubRef_; }
    bool valid() const { return valid_; }
    int bend() const { return bend_; }

    // outer tracker dtc routing block id [0-1]
    int blockId() const;

    // outer tracker dtc routing block channel id [0-35]
    int channelId() const;

    // returns bit accurate representation of Stub
    TTDTC::BV frame(const int& region) const;

    // checks stubs region assignment
    bool inRegion(const int& region) const;

  private:
    // truncates double precision to f/w integer equivalent
    double digi(const double& value, const double& precision) const;

    // returns 64 bit stub in hybrid data format
    TTDTC::BV formatHybrid(const int& region) const;

    // returns 64 bit stub in tmtt data format
    TTDTC::BV formatTMTT(const int& region) const;

  private:
    Settings* settings_;
    TTStubRef ttStubRef_;
    Module* module_;

    bool valid_;                         // passes pt and eta cut
    int col_;                            // column number
    int row_;                            // row number
    int bend_;                           // bend number
    int rowLUT_;                         // reduced row number for look up
    int rowSub_;                         // sub row number inside reduced row number
    double r_;                           // stub r w.r.t. an offset in cm
    double phi_;                         // stub phi w.r.t. detector region centre in rad
    double z_;                           // stub z w.r.t. an offset in cm
    double m_;                           // slope of linearized stub phi in rad / strip
    double c_;                           // intercept of linearized stub phi in rad
    double d_;                           // CIC r in cm
    std::pair<double, double> qOverPt_;  // range of stub qOverPt in 1/cm
    std::pair<double, double> cot_;      // range of stub cot(theta)
    std::pair<double, double> phiT_;     // range of stub extrapolated phi to radius chosenRofPhi in rad

    std::vector<int> regions_;  // shared regions this stub belongs to [0-1]
  };

}  // namespace L1TTrackerDTC

#endif