#include "L1Trigger/TrackFindingTracklet/interface/TrackFitMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackFitMemory::TrackFitMemory(string name, Settings const& settings, unsigned int iSector, double phimin, double phimax)
    : MemoryBase(name, settings, iSector) {
  phimin_ = phimin;
  phimax_ = phimax;
}

void TrackFitMemory::writeTF(bool first) {
  const string dirFT = settings_.memPath() + "FitTrack/";

  std::ostringstream oss;
  oss << dirFT << "TrackFit_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;

    if (not std::filesystem::exists(dirFT)) {
      int fail = system((string("mkdir -p ") + dirFT).c_str());
      if (fail)
        throw cms::Exception("BadDir") << __FILE__ << " " << __LINE__ << " could not create directory " << dirFT;
    }
    out_.open(fname);
    if (out_.fail())
      throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;

  } else
    out_.open(fname, std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracks_.size(); j++) {
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec << " ";
    out_ << tracks_[j]->trackfitstr() << " " << trklet::hexFormat(tracks_[j]->trackfitstr());
    out_ << "\n";
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
