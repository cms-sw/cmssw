#ifndef L1Trigger_TrackerDTC_StubGL_h
#define L1Trigger_TrackerDTC_StubGL_h

#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
#include "L1Trigger/TrackerDTC/interface/StubFE.h"

namespace trackerDTC {

  /*! \class  trackerDTC::StubGL
   *  \brief  Class to represent an Stub transformed to global coordinates by DTC emulator
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class StubGL {
  public:
    StubGL(const StubFE&);
    ~StubGL() = default;
    // underlying front end stub
    const StubFE* stubFE() const { return stubFE_; }
    // event number
    int bx() const { return stubFE_->bx(); }
    // column number in pitch units
    double col() const { return col_; }
    // fine local row number in pitch units wrt fec
    double row() const { return row_; }
    // encoded bend
    double bend() const { return bend_; }
    // rough global row for look up in pitch units
    double fec() const { return fec_; }
    // passes pt and eta cut
    bool valid() const { return valid_; }
    // stub r w.r.t. an offset in cm
    double r() const { return r_; }
    // stub phi w.r.t. detector region centre in rad
    double phi() const { return phi_; }
    // stub z w.r.t. an offset in cm
    double z() const { return z_; }
    // shared regions this stub belongs to [0-1]
    const TTBV& overlap() const { return overlap_; }

  private:
    // underlying front end stub
    const StubFE* stubFE_;
    // column number in pitch units
    double col_;
    // fine local row number in pitch units wrt fec
    double row_;
    // encoded bend
    double bend_;
    // rough global row for look up in pitch units
    double fec_;
    // passes pt and eta cut
    bool valid_;
    // stub r w.r.t. an offset in cm
    double r_;
    // stub phi w.r.t. detector region centre in rad
    double phi_;
    // stub z w.r.t. an offset in cm
    double z_;
    // shared regions this stub belongs to [0-1]
    TTBV overlap_;
  };

}  // namespace trackerDTC

#endif
