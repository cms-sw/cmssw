#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackletParametersMemory::TrackletParametersMemory(string name, Settings const& settings) : MemoryBase(name, settings) {
  npage_ = name.size() - 9;
  tracklets_.resize(npage_);
}

void TrackletParametersMemory::clean() {
  //This is where we delete the tracklets that were created. As tracklet as stored in both the TPAR and MPAR memories
  //we will onlu delete once in the TPAR memory
  if (name_[0] == 'T') {
    for (unsigned int page = 0; page < npage_; page++) {
      for (auto& tracklet : tracklets_[page]) {
        delete tracklet;
      }
    }
  }
  for (unsigned int page = 0; page < tracklets_.size(); page++) {
    tracklets_[page].clear();
  }
}

void TrackletParametersMemory::writeTPAR(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirTP = settings_.memPath() + "TrackletParameters/";

  std::ostringstream oss;
  oss << dirTP << "TrackletParameters_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1)
      << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirTP, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int page = 0; page < tracklets_.size(); page++) {
    for (unsigned int j = 0; j < tracklets_[page].size(); j++) {
      string tpar = tracklets_[page][j]->trackletparstr();
      out_ << hexstr(page) << " " << hexstr(j) << " " << tpar << " " << trklet::hexFormat(tpar) << endl;
    }
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
