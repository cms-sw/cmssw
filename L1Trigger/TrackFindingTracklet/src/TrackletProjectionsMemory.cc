#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackletProjectionsMemory::TrackletProjectionsMemory(string name, Settings const& settings)
    : MemoryBase(name, settings) {
  size_t pos = find_nth(name, 0, "_", 1);
  assert(pos != string::npos);
  initLayerDisk(pos + 1, layer_, disk_);
}

void TrackletProjectionsMemory::addProj(Tracklet* tracklet) {
  if (layer_ != 0 && disk_ == 0)
    assert(tracklet->validProj(layer_ - 1));
  if (layer_ == 0 && disk_ != 0)
    assert(tracklet->validProj(N_LAYER + abs(disk_) - 1));
  if (layer_ != 0 && disk_ != 0)
    assert(tracklet->validProj(layer_ - 1) || tracklet->validProj(N_LAYER + abs(disk_) - 1));

  for (auto& itracklet : tracklets_) {
    if (itracklet == tracklet) {
      edm::LogPrint("Tracklet") << "Adding same tracklet " << tracklet << " twice in " << getName();
    }
    assert(itracklet != tracklet);
  }

  tracklets_.push_back(tracklet);
}

void TrackletProjectionsMemory::clean() { tracklets_.clear(); }

void TrackletProjectionsMemory::writeTPROJ(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirTP = settings_.memPath() + "TrackletProjections/";

  std::ostringstream oss;
  oss << dirTP << "TrackletProjections_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1)
      << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirTP, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    string proj = (layer_ > 0 && tracklets_[j]->validProj(layer_ - 1)) ? tracklets_[j]->trackletprojstrlayer(layer_)
                                                                       : tracklets_[j]->trackletprojstrdisk(disk_);
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
