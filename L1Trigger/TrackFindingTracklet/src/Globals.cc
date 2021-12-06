// Globals: holds "global" variables such as the IMATH_TrackletCalculators
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/imath.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorOverlap.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

using namespace std;
using namespace trklet;

Globals::Globals(Settings const& settings) {
  imathGlobals* imathGlobs = new imathGlobals();

  //takes owernship of globals pointer
  imathGlobals_.reset(imathGlobs);

  // tracklet calculators
  ITC_L1L2_ = make_unique<IMATH_TrackletCalculator>(settings, imathGlobs, 1, 2);
  ITC_L2L3_ = make_unique<IMATH_TrackletCalculator>(settings, imathGlobs, 2, 3);
  ITC_L3L4_ = make_unique<IMATH_TrackletCalculator>(settings, imathGlobs, 3, 4);
  ITC_L5L6_ = make_unique<IMATH_TrackletCalculator>(settings, imathGlobs, 5, 6);

  ITC_F1F2_ = make_unique<IMATH_TrackletCalculatorDisk>(settings, imathGlobs, 1, 2);
  ITC_F3F4_ = make_unique<IMATH_TrackletCalculatorDisk>(settings, imathGlobs, 3, 4);
  ITC_B1B2_ = make_unique<IMATH_TrackletCalculatorDisk>(settings, imathGlobs, -1, -2);
  ITC_B3B4_ = make_unique<IMATH_TrackletCalculatorDisk>(settings, imathGlobs, -3, -4);

  ITC_L1F1_ = make_unique<IMATH_TrackletCalculatorOverlap>(settings, imathGlobs, 1, 1);
  ITC_L2F1_ = make_unique<IMATH_TrackletCalculatorOverlap>(settings, imathGlobs, 2, 1);
  ITC_L1B1_ = make_unique<IMATH_TrackletCalculatorOverlap>(settings, imathGlobs, 1, -1);
  ITC_L2B1_ = make_unique<IMATH_TrackletCalculatorOverlap>(settings, imathGlobs, 2, -1);
}

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
