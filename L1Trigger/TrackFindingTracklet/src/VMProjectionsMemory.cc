#include "L1Trigger/TrackFindingTracklet/interface/VMProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

VMProjectionsMemory::VMProjectionsMemory(string name, Settings const& settings) : MemoryBase(name, settings) {
  initLayerDisk(7, layer_, disk_);
}

void VMProjectionsMemory::addTracklet(Tracklet* tracklet, unsigned int allprojindex) {
  std::pair<Tracklet*, unsigned int> tmp(tracklet, allprojindex);
  //Check that order of TCID is correct
  if (!tracklets_.empty()) {
    assert(tracklets_[tracklets_.size() - 1].first->TCID() <= tracklet->TCID());
  }
  tracklets_.push_back(tmp);
}

void VMProjectionsMemory::writeVMPROJ(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirVM = settings_.memPath() + "VMProjections/";

  std::ostringstream oss;
  oss << dirVM + "VMProjections_" << getName();
  //get rid of duplicates
  auto const& tmp = oss.str();
  int len = tmp.size();
  if (tmp[len - 2] == 'n' && tmp[len - 1] > '1' && tmp[len - 1] <= '9')
    return;
  oss << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirVM, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    string vmproj = (layer_ > 0) ? tracklets_[j].first->vmstrlayer(layer_, tracklets_[j].second)
                                 : tracklets_[j].first->vmstrdisk(disk_, tracklets_[j].second);
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << vmproj << " " << trklet::hexFormat(vmproj) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
