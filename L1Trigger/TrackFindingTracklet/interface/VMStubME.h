#ifndef L1Trigger_TrackFindingTracklet_interface_VMStubME_h
#define L1Trigger_TrackFindingTracklet_interface_VMStubME_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

namespace trklet {

  class VMStubME {
  public:
    VMStubME() {}

    VMStubME(const Stub* stub, FPGAWord finephi, FPGAWord finerz, FPGAWord bend, FPGAWord allstubindex);

    ~VMStubME() = default;

    const FPGAWord& finephi() const { return finephi_; }
    const FPGAWord& finerz() const { return finerz_; }

    const FPGAWord& bend() const { return bend_; }

    const Stub* stub() const { return stub_; }

    bool isPSmodule() const { return stub_->isPSmodule(); }

    const FPGAWord& stubindex() const { return allStubIndex_; }

    //return binary string for memory printout
    std::string str() const;

  private:
    FPGAWord allStubIndex_;
    FPGAWord finephi_;
    FPGAWord finerz_;
    FPGAWord bend_;
    const Stub* stub_;
  };

};  // namespace trklet
#endif
