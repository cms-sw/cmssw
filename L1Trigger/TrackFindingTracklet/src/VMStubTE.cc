#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"

using namespace std;
using namespace trklet;

VMStubTE::VMStubTE(const Stub* stub, FPGAWord finephi, FPGAWord bend, FPGAWord vmbits, FPGAWord allstubindex) {
  stub_ = stub;
  finephi_ = finephi;
  bend_ = bend;
  vmbits_ = vmbits;
  allStubIndex_ = allstubindex;
}

std::string VMStubTE::str() const {
  string stub = allStubIndex_.str();
  stub += "|";
  stub += bend_.str();
  stub += "|";
  stub += finephi_.str();
  stub += "|";
  stub += vmbits_.str();

  return stub;
}
