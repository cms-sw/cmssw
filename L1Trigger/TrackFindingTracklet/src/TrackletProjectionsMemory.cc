#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include <iomanip>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

TrackletProjectionsMemory::TrackletProjectionsMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {
  if (settings_.extended()) {
    initLayerDisk(14, layer_, disk_);
  } else {
    initLayerDisk(12, layer_, disk_);
  }
}

void TrackletProjectionsMemory::addProj(Tracklet* tracklet) {
  if (layer_ != 0 && disk_ == 0)
    assert(tracklet->validProj(layer_));
  if (layer_ == 0 && disk_ != 0)
    assert(tracklet->validProjDisk(disk_));
  if (layer_ != 0 && disk_ != 0)
    assert(tracklet->validProj(layer_) || tracklet->validProjDisk(disk_));

  for (auto& itracklet : tracklets_) {
    if (itracklet == tracklet) {
      edm::LogPrint("Tracklet") << "Adding same tracklet " << tracklet << " twice in " << getName();
    }
    assert(itracklet != tracklet);
  }

  tracklets_.push_back(tracklet);
}

void TrackletProjectionsMemory::clean() { tracklets_.clear(); }

void TrackletProjectionsMemory::writeTPROJ(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/TrackletProjections/TrackletProjections_" << getName() << "_" << std::setfill('0')
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
    string proj = (layer_ > 0 && tracklets_[j]->validProj(layer_)) ? tracklets_[j]->trackletprojstrlayer(layer_)
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
