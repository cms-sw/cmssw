#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackletParametersMemory::TrackletParametersMemory(string name, Settings const& settings)
    : MemoryBase(name, settings) {}

void TrackletParametersMemory::clean() {
  for (auto& tracklet : tracklets_) {
    delete tracklet;
  }
  tracklets_.clear();
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

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    string tpar = tracklets_[j]->trackletparstr();
    out_ << hexstr(j) << " " << tpar << " " << trklet::hexFormat(tpar) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
