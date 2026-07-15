// Globals: holds "global" variables such as the IMATH_TrackletCalculators
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

using namespace std;
using namespace trklet;

Globals::Globals(Settings const& settings) {}

Globals::~Globals() {
  for (auto i : thePhiCorr_) {
    delete i;
    i = nullptr;
  }
}

std::ofstream& Globals::ofstream(std::string fname) {
  if (ofstreams_.find(fname) != ofstreams_.end()) {
    return *(ofstreams_[fname]);
  }
  std::ofstream* outptr = new std::ofstream(fname.c_str());
  ofstreams_[fname] = outptr;
  return *outptr;
}
