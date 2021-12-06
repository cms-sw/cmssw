#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include <iomanip>
#include <cmath>
#include <sstream>
#include <cctype>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace trklet;
using namespace std;

InputLinkMemory::InputLinkMemory(string name, Settings const& settings, double, double) : MemoryBase(name, settings) {}

void InputLinkMemory::addStub(Stub* stub) { stubs_.push_back(stub); }

void InputLinkMemory::writeStubs(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirIS = settings_.memPath() + "InputStubs/";
  openFile(first, dirIS, "InputStubs_");

  for (unsigned int j = 0; j < stubs_.size(); j++) {
    string stub = stubs_[j]->str();
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << stub << " " << trklet::hexFormat(stub) << endl;
  }
  out_.close();
}

void InputLinkMemory::clean() { stubs_.clear(); }
