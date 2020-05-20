#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"
#include <iomanip>

using namespace std;
using namespace trklet;

StubPairsMemory::StubPairsMemory(string name, const Settings* const settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {}

void StubPairsMemory::writeSP(bool first) {
  std::ostringstream oss;
  oss << "../data/MemPrints/StubPairs/StubPairs_" << getName() << "_" << std::setfill('0') << std::setw(2)
      << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;
    out_.open(fname.c_str());
  } else
    out_.open(fname.c_str(), std::ofstream::app);

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
