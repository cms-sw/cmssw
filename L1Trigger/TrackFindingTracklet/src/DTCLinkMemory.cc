#include "L1Trigger/TrackFindingTracklet/interface/DTCLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"

#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include <iomanip>
#include <cmath>
#include <sstream>
#include <cctype>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace trklet;
using namespace std;

DTCLinkMemory::DTCLinkMemory(string name, Settings const& settings, double, double) : MemoryBase(name, settings) {}

void DTCLinkMemory::addStub(const L1TStub& al1stub, const Stub& stub) {
  //Make new objects owned by the dtclink memory and save in list of stubs
  if (stubs_.size() < settings_.maxStep("IR")) {
    Stub* stubptr = new Stub(stub);
    stubptr->setl1tstub(new L1TStub(al1stub));

    stubs_.emplace_back(stubptr);
  }
}

void DTCLinkMemory::writeStubs(bool first, unsigned int iSector) {
  iSector_ = iSector;

  const string dirIS = settings_.memPath() + "InputStubs/";
  openFile(first, dirIS, "Link_");

  for (unsigned int j = 0; j < stubs_.size(); j++) {
    string dtcname = stubs_[j]->l1tstub()->DTClink();
    int layerdisk = stubs_[j]->l1tstub()->layerdisk();

    //If the string starts with 'neg' skip the first three character
    int start = dtcname.substr(0, 3) == "neg" ? 3 : 0;

    //For the dtcbase name remove the leading 'neg' if in the name and the trailing '_A' or '_B'
    string dtcbase = dtcname.substr(start, dtcname.size() - 2 - start);

    const vector<int>& layers = settings_.dtcLayers(dtcbase);

    int lcode = -1;
    for (unsigned int index = 0; index < layers.size(); index++) {
      if (layerdisk == layers[index]) {
        lcode = index;
      }
    }
    assert(lcode != -1);

    FPGAWord ldcode(lcode, 2, true);

    string stub = stubs_[j]->str() + "|" + ldcode.str() + "|1";
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << stub << " " << trklet::hexFormat(stub) << endl;
  }
  out_.close();
}

void DTCLinkMemory::clean() {
  for (auto& stub : stubs_) {
    delete stub->l1tstub();
    delete stub;
  }
  stubs_.clear();
}
