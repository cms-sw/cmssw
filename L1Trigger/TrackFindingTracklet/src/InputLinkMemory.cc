#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterPhiCorrTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include <iomanip>
#include <cmath>
#include <sstream>
#include <cctype>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace trklet;
using namespace std;

InputLinkMemory::InputLinkMemory(string name, Settings const& settings, unsigned int iSector, double, double)
    : MemoryBase(name, settings, iSector) {
  string subname = name.substr(5, 7);
  phiregion_ = subname[3] - 'A';
  assert(phiregion_ >= 0 && phiregion_ < 8);

  layerdisk_ = initLayerDisk(3);
}

bool InputLinkMemory::addStub(
    Settings const& settings, Globals* globals, L1TStub& al1stub, Stub& stub, string dtc = "") {
  if (layerdisk_ < N_LAYER && globals->phiCorr(layerdisk_) == nullptr) {
    globals->phiCorr(layerdisk_) = new VMRouterPhiCorrTable(settings);
    int nbits = 3;
    if (layerdisk_ >= N_PSLAYER)
      nbits = 4;
    globals->phiCorr(layerdisk_)->init(layerdisk_ + 1, nbits, 3);
  }

  unsigned int stublayerdisk = stub.layerdisk();
  assert(stublayerdisk < 11);

  if (stublayerdisk != layerdisk_)
    return false;

  if (layerdisk_ < N_LAYER) {
    FPGAWord r = stub.r();
    int bendbin = stub.bend().value();
    int rbin = (r.value() + (1 << (r.nbits() - 1))) >> (r.nbits() - 3);
    const VMRouterPhiCorrTable& phiCorrTable = *globals->phiCorr(layerdisk_);
    int iphicorr = phiCorrTable.getphiCorrValue(bendbin, rbin);
    stub.setPhiCorr(iphicorr);
  }

  FPGAWord iphi = stub.phicorr();
  unsigned int nallbits = settings_.nbitsallstubs(layerdisk_);
  int phibin = iphi.bits(iphi.nbits() - nallbits, nallbits);
  int iphivmRaw = iphi.bits(iphi.nbits() - 5, 5);

  if (phibin != phiregion_)
    return false;

  if (getName().substr(10, dtc.size()) != dtc)
    return false;

  string half = getName().substr(getName().size() - 3, 3);
  if (half[1] != 'n') {
    half = getName().substr(getName().size() - 1, 1);
  }

  assert(half[0] == 'A' || half[0] == 'B');

  if (half[0] == 'B' && iphivmRaw <= 15)
    return false;
  if (half[0] == 'A' && iphivmRaw > 15)
    return false;

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Will add stub in " << getName() << " "
                                 << "iphiwmRaw = " << iphivmRaw << " phi=" << al1stub.phi() << " z=" << al1stub.z()
                                 << " r=" << al1stub.r();
  }
  if (stubs_.size() < settings_.maxStep("Link")) {
    Stub* stubptr = new Stub(stub);
    stubptr->setl1tstub(new L1TStub(al1stub));

    stubs_.emplace_back(stubptr);
  }
  return true;
}

void InputLinkMemory::writeStubs(bool first) {
  openFile(first, "../data/MemPrints/InputStubs/InputStubs_");

  for (unsigned int j = 0; j < stubs_.size(); j++) {
    string stub = stubs_[j]->str();
    out_ << std::setfill('0') << std::setw(2);
    out_ << hex << j << dec;
    out_ << " " << stub << " " << trklet::hexFormat(stub) << endl;
  }
  out_.close();
}

void InputLinkMemory::clean() {
  for (auto& stub : stubs_) {
    delete stub->l1tstub();
    delete stub;
  }
  stubs_.clear();
}
