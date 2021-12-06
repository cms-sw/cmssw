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

  std::ostringstream oss;
  oss << dirCT << "CleanTrack_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirCT, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracks_.size(); j++) {
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec << " ";
    out_ << tracks_[j]->trackfitstr() << " " << trklet::hexFormat(tracks_[j]->trackfitstr());
    out_ << "\n";
  }
  out_.close();

  // --------------------------------------------------------------
  // print separately ALL cleaned tracks in single file
  if (settings_.writeMonitorData("CT")) {
    std::string fnameAll = "CleanTracksAll.dat";
    if (first && getName() == "CT_L1L2" && iSector_ == 0)
      out_.open(fnameAll);
    else
      out_.open(fnameAll, std::ofstream::app);

    if (!tracks_.empty())
      out_ << "BX= " << (bitset<3>)bx_ << " event= " << event_ << " seed= " << getName()
           << " phisector= " << iSector_ + 1 << endl;

    for (unsigned int j = 0; j < tracks_.size(); j++) {
      if (j < 16)
        out_ << "0";
      out_ << hex << j << dec << " ";
      out_ << tracks_[j]->trackfitstr();
      out_ << "\n";
    }
    out_.close();
  }
  // --------------------------------------------------------------

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
