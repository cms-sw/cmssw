#ifndef L1Trigger_TrackFindingTracklet_interface_VMStubTE_h
#define L1Trigger_TrackFindingTracklet_interface_VMStubTE_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

namespace trklet {

  class VMStubTE {
  public:
    VMStubTE() {}

    VMStubTE(const Stub* stub, FPGAWord finephi, FPGAWord bend, FPGAWord vmbits, FPGAWord allstubindex);

    ~VMStubTE() = default;

    const FPGAWord& finephi() const { return finephi_; }

    const FPGAWord& bend() const { return bend_; }

    const FPGAWord& vmbits() const { return vmbits_; }

    const Stub* stub() const { return stub_; }

    bool isPSmodule() const { return stub_->isPSmodule(); }

    const FPGAWord& stubindex() const { return allStubIndex_; }

    //return binary string for memory printout
    std::string str() const;

  private:
    FPGAWord finephi_;
    FPGAWord bend_;
    FPGAWord vmbits_;
    FPGAWord allStubIndex_;
    const Stub* stub_;
  };
};  // namespace trklet
#endif
