// Globals: holds "global" variables such as the IMATH_TrackletCalculators
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/imath.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculatorOverlap.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterPhiCorrTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#ifdef USEHYBRID
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#endif

using namespace std;
using namespace trklet;

Globals::Globals(Settings const& settings) {
  imathGlobals* imathGlobs = new imathGlobals();

  //takes owernship of globals pointer
  imathGlobals_.reset(imathGlobs);

  // tracklet calculators
  ITC_L1L2_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 1, 2));
  ITC_L2L3_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 2, 3));
  ITC_L3L4_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 3, 4));
  ITC_L5L6_.reset(new IMATH_TrackletCalculator(settings, imathGlobs, 5, 6));

  ITC_F1F2_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, 1, 2));
  ITC_F3F4_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, 3, 4));
  ITC_B1B2_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, -1, -2));
  ITC_B3B4_.reset(new IMATH_TrackletCalculatorDisk(settings, imathGlobs, -3, -4));

  ITC_L1F1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 1, 1));
  ITC_L2F1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 2, 1));
  ITC_L1B1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 1, -1));
  ITC_L2B1_.reset(new IMATH_TrackletCalculatorOverlap(settings, imathGlobs, 2, -1));
}

Globals::~Globals() {
  for (auto i : thePhiCorr_) {
    delete i;
    i = nullptr;
  }
#ifdef USEHYBRID
  delete tmttSettings_;
  tmttSettings_ = nullptr;
  delete tmttKFParamsComb_;
  tmttKFParamsComb_ = nullptr;
#endif
}

std::ofstream& Globals::ofstream(std::string fname) {
  if (ofstreams_.find(fname) != ofstreams_.end()) {
    return *(ofstreams_[fname]);
  }
  std::ofstream* outptr = new std::ofstream(fname.c_str());
  ofstreams_[fname] = outptr;
  return *outptr;
}
