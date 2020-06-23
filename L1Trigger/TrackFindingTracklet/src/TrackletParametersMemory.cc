#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>

using namespace std;
using namespace trklet;

TrackletParametersMemory::TrackletParametersMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {}

void TrackletParametersMemory::clean() {
  for (auto& tracklet : tracklets_) {
    delete tracklet;
  }
  tracklets_.clear();
}

void TrackletParametersMemory::writeMatches(Globals* globals, int& matchesL1, int& matchesL3, int& matchesL5) {
  ofstream& out = globals->ofstream("nmatches.txt");
  for (auto& tracklet : tracklets_) {
    if ((tracklet->nMatches() + tracklet->nMatchesDisk()) > 0) {
      if (tracklet->layer() == 1)
        matchesL1++;
      if (tracklet->layer() == 3)
        matchesL3++;
      if (tracklet->layer() == 5)
        matchesL5++;
    }
    out << tracklet->layer() << " " << tracklet->disk() << " " << tracklet->nMatches() << " "
        << tracklet->nMatchesDisk() << endl;
  }
}

void TrackletParametersMemory::writeTPAR(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/TrackletParameters/TrackletParameters_" << getName() << "_" << std::setfill('0')
      << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;
    out_.open(fname.c_str());
  } else
    out_.open(fname.c_str(), std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    string tpar = tracklets_[j]->trackletparstr();
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << tpar << " " << trklet::hexFormat(tpar) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
