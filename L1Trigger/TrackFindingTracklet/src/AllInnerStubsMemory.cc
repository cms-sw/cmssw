#include "L1Trigger/TrackFindingTracklet/interface/AllInnerStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include <iomanip>

using namespace std;
using namespace trklet;

AllInnerStubsMemory::AllInnerStubsMemory(string name, Settings const& settings) : MemoryBase(name, settings) {}

void AllInnerStubsMemory::writeStubs(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirS = settings_.memPath() + "Stubs/";
  openFile(first, dirS, "AllInnerStubs_");

  for (unsigned int j = 0; j < stubs_.size(); j++) {
    string stub = stubs_[j]->strinner();
    out_ << hexstr(j) << " " << stub << " " << hexFormat(stub) << endl;
  }
  out_.close();
}
