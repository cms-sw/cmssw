#include "L1Trigger/TrackFindingTracklet/interface/FullMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include <iomanip>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

FullMatchMemory::FullMatchMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {
  if (settings_.extended()) {
    initLayerDisk(10, layer_, disk_);
  } else {
    initLayerDisk(8, layer_, disk_);
  }
}

void FullMatchMemory::addMatch(Tracklet* tracklet, const Stub* stub) {
  if (!settings_.doKF()) {  //When using KF we allow multiple matches
    for (auto& match : matches_) {
      if (match.first == tracklet) {  //Better match, replace
        match.second = stub;
        return;
      }
    }
  }
  std::pair<Tracklet*, const Stub*> tmp(tracklet, stub);
  //Check that we have the right TCID order
  if (!matches_.empty()) {
    if ((!settings_.doKF() && matches_[matches_.size() - 1].first->TCID() >= tracklet->TCID()) ||
        (settings_.doKF() && matches_[matches_.size() - 1].first->TCID() > tracklet->TCID())) {
      edm::LogPrint("Tracklet") << "Wrong TCID ordering in " << getName() << " : "
                                << matches_[matches_.size() - 1].first->TCID() << " " << tracklet->TCID() << " "
                                << matches_[matches_.size() - 1].first->trackletIndex() << " "
                                << tracklet->trackletIndex();
    }
  }
  matches_.push_back(tmp);
}

void FullMatchMemory::writeMC(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/Matches/FullMatches_" << getName() << "_" << std::setfill('0') << std::setw(2)
      << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;
    out_.open(fname.c_str());
  } else
    out_.open(fname.c_str(), std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < matches_.size(); j++) {
    string match = (layer_ > 0) ? matches_[j].first->fullmatchstr(layer_) : matches_[j].first->fullmatchdiskstr(disk_);
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << match << " " << trklet::hexFormat(match) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
