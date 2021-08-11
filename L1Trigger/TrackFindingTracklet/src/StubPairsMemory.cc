#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

StubPairsMemory::StubPairsMemory(string name, Settings const& settings) : MemoryBase(name, settings) {}

void StubPairsMemory::writeSP(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirSP = settings_.memPath() + "StubPairs/";

  std::ostringstream oss;
  oss << dirSP << "StubPairs_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirSP, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < stubs_.size(); j++) {
    string stub1index = stubs_[j].first.stub()->stubindex().str();
    string stub2index = stubs_[j].second.stub()->stubindex().str();
    out_ << "0x";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << stub1index << "|" << stub2index << " " << trklet::hexFormat(stub1index + stub2index) << endl;
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}
