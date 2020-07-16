#include "L1Trigger/TrackFindingTracklet/interface/StubTripletsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include <iomanip>

using namespace std;
using namespace trklet;

StubTripletsMemory::StubTripletsMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {}

void StubTripletsMemory::writeST(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/StubPairs/StubTriplets_" << getName() << "_" << std::setfill('0') << std::setw(2)
      << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;
    out_.open(fname.c_str());
  } else
    out_.open(fname.c_str(), std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < stubs1_.size(); j++) {
    string stub1index = stubs1_[j]->stubindex().str();
    string stub2index = stubs2_[j]->stubindex().str();
    string stub3index = stubs3_[j]->stubindex().str();
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << stub1index << "|" << stub2index << "|" << stub3index << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
