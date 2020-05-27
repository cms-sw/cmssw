#include "L1Trigger/TrackFindingTracklet/interface/VMStubME.h"

using namespace std;
using namespace trklet;

VMStubME::VMStubME(const Stub* stub, FPGAWord finephi, FPGAWord finerz, FPGAWord bend, FPGAWord allstubindex) {
  stub_ = stub;
  finephi_ = finephi;
  finerz_ = finerz;
  bend_ = bend;
  allStubIndex_ = allstubindex;
}

std::string VMStubME::str() const {
  string stub = allStubIndex_.str();
  stub += "|";
  stub += bend_.str();
  stub += "|";
  stub += finephi_.str();
  stub += "|";
  stub += finerz_.str();

  return stub;
}
