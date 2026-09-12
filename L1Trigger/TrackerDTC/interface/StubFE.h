#ifndef L1Trigger_TrackerDTC_StubFE_h
#define L1Trigger_TrackerDTC_StubFE_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"

namespace trackerDTC {

  /*! \class  trackerDTC::StubFE
   *  \brief  Class to represent an outer tracker Stub before its processing in DTC
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class StubFE {
  public:
    StubFE(const Setup*, const SensorModule*, const TTStubRef&, int = 0);
    StubFE(const Setup*, int, int, TTBV&);
    ~StubFE() = default;
    // stores, calculates and provides run-time constants
    const Setup* setup() const { return setup_; }
    // representation of an outer tracker sensormodule
    const SensorModule* sm() const { return sm_; }
    // underlying TTStubRef
    const TTStubRef& ttStubRef() const { return ttStubRef_; }
    // event number
    int bx() const { return bx_; }
    // column number in pitch units wrt cic
    int col() const { return col_; }
    // row number in half pitch units wrt fec
    int row() const { return row_; }
    // encoded bend
    int bend() const { return bend_; }
    // front end (CBC or MPA) identifier
    int fec() const { return fec_; }
    // front end (CBC or MPA) identifier
    int cic() const { return cic_; }
    // sensor module identifier
    int channel() const { return channel_; }
    // bit accurate representation of Stub
    const TTBV& ttBV() const { return ttBV_; }
    // for std::find
    bool operator==(const StubFE&) const;

  private:
    // stores, calculates and provides run-time constants
    const Setup* setup_;
    // representation of an outer tracker sensormodule
    const SensorModule* sm_;
    // underlying TTStubRef
    TTStubRef ttStubRef_;
    // event number
    int bx_;
    // column number in pitch units wrt cic
    int col_;
    // row number in half pitch units wrt fec
    int row_;
    // encoded bend
    int bend_;
    // front end chip (CBC or MPA) identifier
    int fec_;
    // front end cic identifier
    int cic_;
    // sensor module identifier
    int channel_;
    // bit accurate representation of Stub
    TTBV ttBV_;
  };

}  // namespace trackerDTC

#endif
