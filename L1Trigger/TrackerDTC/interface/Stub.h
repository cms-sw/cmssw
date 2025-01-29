#ifndef L1Trigger_TrackerDTC_Stub_h
#define L1Trigger_TrackerDTC_Stub_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <utility>
#include <vector>

namespace trackerDTC {

  /*! \class  trackerDTC::Stub
   *  \brief  Class to represent an outer tracker Stub
   *  \author Thomas Schuh
   *  \date   2020, Jan
   */
  class Stub {
  public:
    Stub(const trackerTFP::DataFormats*, const tt::SensorModule*, const TTStubRef&);
    Stub(const tt::Setup*,
         const trackerTFP::DataFormats*,
         const LayerEncoding*,
         const tt::SensorModule*,
         const TTStubRef&);
    ~Stub() {}
    // underlying TTStubRef
    const TTStubRef& ttStubRef() const { return ttStubRef_; }
    // did pass pt and eta cut
    bool valid() const { return valid_; }
    // stub bend in quarter pitch units
    int bend() const { return bend_; }
    // bit accurate representation of Stub
    tt::FrameStub frame(int region) const;
    // checks stubs region assignment
    bool inRegion(int region) const { return regions_[region]; }
    // range of stub extrapolated phi to radius chosenRofPhi in rad
    std::pair<double, double> phiT() const { return phiT_; }
    // stub phi w.r.t. detector region centre in rad
    double phi() const { return phi_; }

  private:
    // truncates double precision to f/w integer equivalent
    double digi(double value, double precision) const;
    // 64 bit stub in hybrid data format
    tt::Frame formatHybrid(int region) const;
    // 64 bit stub in tmtt data format
    tt::Frame formatTMTT(int region) const;
    // stores, calculates and provides run-time constants
    const tt::Setup* setup_;
    // helper class to extract structured data from tt::Frames
    const trackerTFP::DataFormats* dataFormats_;
    // class to encode layer ids used between DTC and TFP in Hybrid
    const LayerEncoding* layerEncoding_;
    // representation of an outer tracker sensormodule
    const tt::SensorModule* sm_;
    // underlying TTStubRef
    const TTStubRef ttStubRef_;
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
    // range of stub inv2R in 1/cm
    std::pair<double, double> inv2R_;
    // range of stub extrapolated phi to radius chosenRofPhi wrt detector nonant center in rad
    std::pair<double, double> phiT_;
    // range of stub extrapolated z to radius chosenRofZ in cm
    std::pair<double, double> zT_;
    // shared regions this stub belongs to [0-1]
    TTBV regions_;
  };

}  // namespace trackerDTC

#endif
