#include "L1Trigger/TrackFindingTracklet/interface/CleanTrackMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include <iomanip>

using namespace std;
using namespace trklet;

CleanTrackMemory::CleanTrackMemory(
    string name, const Settings* const settings, unsigned int iSector, double phimin, double phimax)
    : MemoryBase(name, settings, iSector) {
  phimin_ = phimin;
  phimax_ = phimax;
}

void CleanTrackMemory::writeCT(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/CleanTrack/CleanTrack_" << getName() << "_" << std::setfill('0') << std::setw(2)
      << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;
    out_.open(fname.c_str());
  } else
    out_.open(fname.c_str(), std::ofstream::app);

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
  if (settings_->writeMonitorData("CT")) {
    std::string fnameAll = "CleanTracksAll.dat";
    if (first && getName() == "CT_L1L2" && iSector_ == 0)
      out_.open(fnameAll.c_str());
    else
      out_.open(fnameAll.c_str(), std::ofstream::app);

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
