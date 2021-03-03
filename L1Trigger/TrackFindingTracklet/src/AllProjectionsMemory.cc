#include "L1Trigger/TrackFindingTracklet/interface/AllProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>
#include <filesystem>

using namespace trklet;
using namespace std;

AllProjectionsMemory::AllProjectionsMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {
  initLayerDisk(3, layer_, disk_);
}

void AllProjectionsMemory::writeAP(bool first) {
  const string dirTP = settings_.memPath() + "TrackletProjections/";

  std::ostringstream oss;
  oss << dirTP << "AllProj_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;

    if (not std::filesystem::exists(dirTP)) {
      int fail = system((string("mkdir -p ") + dirTP).c_str());
      if (fail)
        throw cms::Exception("BadDir") << __FILE__ << " " << __LINE__ << " could not create directory " << dirTP;
    }
    out_.open(fname);
    if (out_.fail())
      throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;

  } else
    out_.open(fname, std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    string proj =
        (layer_ > 0) ? tracklets_[j]->trackletprojstrlayer(layer_) : tracklets_[j]->trackletprojstrdisk(disk_);
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << proj << "  " << trklet::hexFormat(proj) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
