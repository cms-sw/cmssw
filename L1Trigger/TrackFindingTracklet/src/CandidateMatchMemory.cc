#include "L1Trigger/TrackFindingTracklet/interface/CandidateMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

CandidateMatchMemory::CandidateMatchMemory(string name, Settings const& settings) : MemoryBase(name, settings) {}

void CandidateMatchMemory::addMatch(std::pair<Tracklet*, int> tracklet, const Stub* stub) {
  std::pair<std::pair<Tracklet*, int>, const Stub*> tmp(tracklet, stub);

  //Check for consistency
  for (auto& match : matches_) {
    if (tracklet.first->TCID() < match.first.first->TCID()) {
      throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " In " << getName() << " adding tracklet "
                                         << tracklet.first << " with lower TCID : " << tracklet.first->TCID()
                                         << " than earlier TCID " << match.first.first->TCID();
    }
  }
  matches_.push_back(tmp);
}

void CandidateMatchMemory::writeCM(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirM = settings_.memPath() + "Matches/";

  std::ostringstream oss;
  oss << dirM << "CandidateMatches_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1)
      << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirM, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < matches_.size(); j++) {
    string stubid = matches_[j].second->stubindex().str();  // stub ID
    int projindex = matches_[j].first.second;               // Allproj index
    FPGAWord tmp;
    if (projindex >= (1 << 7)) {
      projindex = (1 << 7) - 1;
    }
    tmp.set(projindex, 7, true, __LINE__, __FILE__);
    out_ << hexstr(j) << " " << tmp.str() << "|" << stubid << " " << trklet::hexFormat(tmp.str() + stubid) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
