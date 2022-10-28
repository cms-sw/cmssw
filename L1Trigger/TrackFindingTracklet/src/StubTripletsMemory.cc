#include "L1Trigger/TrackFindingTracklet/interface/StubTripletsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

StubTripletsMemory::StubTripletsMemory(string name, Settings const& settings) : MemoryBase(name, settings) {}

void StubTripletsMemory::writeST(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirSP = settings_.memPath() + "StubPairs/";

  std::ostringstream oss;
  oss << dirSP << "StubTriplets_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirSP, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < stubs1_.size(); j++) {
    string stub1index = stubs1_[j]->stubindex().str();
    string stub2index = stubs2_[j]->stubindex().str();
    string stub3index = stubs3_[j]->stubindex().str();
    out_ << hexstr(j) << " " << stub1index << "|" << stub2index << "|" << stub3index << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
