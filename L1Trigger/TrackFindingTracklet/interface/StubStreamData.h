#ifndef L1Trigger_TrackFindingTracklet_interface_StubStreamData_h
#define L1Trigger_TrackFindingTracklet_interface_StubStreamData_h

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"

#include <string>

// Represents an element of the bit-accurate stub stream from TrackBuilder output
// (This class only needed to support stand-alone running of this code).

namespace trklet {

  class L1TStub;

  class StubStreamData {
  public:
    StubStreamData() {}

    StubStreamData(int iSeed, const L1TStub& stub, const std::string& dataBits)
        : iSeed_(iSeed), stub_(stub), dataBits_(dataBits) {}

    ~StubStreamData() = default;

    int iSeed() const { return iSeed_; }          // Seed type
    bool valid() const { return (iSeed_ >= 0); }  // Valid stub
    const L1TStub& stub() const { return stub_; }
    // String with bits of valid bit + r coordinate + phi residual + r or z residual.
    const std::string& dataBits() const { return dataBits_; }

  private:
    int iSeed_{-1};
    L1TStub stub_;
    std::string dataBits_{""};
  };
};  // namespace trklet
#endif
