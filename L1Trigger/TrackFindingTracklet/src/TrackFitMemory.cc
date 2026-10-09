#include "L1Trigger/TrackFindingTracklet/interface/TrackFitMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackFitMemory::TrackFitMemory(string name, Settings const& settings, double phimin, double phimax)
    : MemoryBase(name, settings) {
  phimin_ = phimin;
  phimax_ = phimax;
}

void TrackFitMemory::writeTF(bool first, unsigned int iSector) {
  iSector_ = iSector;

  const string dirFT = settings_.memPath() + "FitTrack/";
  openFile(first, dirFT, "TrackFit_");

  for (unsigned int j = 0; j < tracks_.size(); j++) {
    out_ << hexstr(j) << " " << tracks_[j]->trackfitstr() << " " << trklet::hexFormat(tracks_[j]->trackfitstr());
    out_ << "\n";
  }
  out_.close();
}
