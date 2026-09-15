#include "L1Trigger/TrackFindingTracklet/interface/CleanTrackMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

CleanTrackMemory::CleanTrackMemory(string name, Settings const& settings, double phimin, double phimax)
    : MemoryBase(name, settings) {
  phimin_ = phimin;
  phimax_ = phimax;
}

void CleanTrackMemory::writeCT(bool first, unsigned int iSector) {
  iSector_ = iSector;

  const string dirCT = settings_.memPath() + "CleanTrack/";
  openFile(first, dirCT, "CleanTrack_");

  for (unsigned int j = 0; j < tracks_.size(); j++) {
    out_ << hexstr(j) << " " << tracks_[j]->trackfitstr() << " " << trklet::hexFormat(tracks_[j]->trackfitstr());
    out_ << "\n";
  }
  out_.close();

  // --------------------------------------------------------------
  // print separately ALL cleaned tracks in single file
  if (settings_.writeMonitorData("CT")) {
    std::string fnameAll = "CleanTracksAll.dat";
    if (first && getName() == "CT_L1L2" && iSector_ == 0)
      outCT_.open(fnameAll);
    else
      outCT_.open(fnameAll, std::ofstream::app);

    if (!tracks_.empty())
      outCT_ << "BX= " << (bitset<3>)bx_ << " event= " << event_ << " seed= " << getName()
             << " phisector= " << iSector_ + 1 << endl;

    for (unsigned int j = 0; j < tracks_.size(); j++) {
      if (j < 16)
        outCT_ << "0";
      outCT_ << hex << j << dec << " ";
      outCT_ << tracks_[j]->trackfitstr();
      outCT_ << "\n";
    }
    outCT_.close();
  }
  // --------------------------------------------------------------
}
