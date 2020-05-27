#include "L1Trigger/TrackFindingTracklet/interface/DTC.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"

#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

DTC::DTC(string name) {
  name_ = name;
  for (unsigned int i = 0; i < N_LAYER + N_DISK; i++) {
    phimin_[i] = 10.0;
    phimax_[i] = -10.0;
  }
}

void DTC::setName(string name) { name_ = name; }

void DTC::addSec(int sector) { sectors_.push_back(sector); }

void DTC::addphi(double phi, unsigned int layerdisk) {
  assert(layerdisk < N_LAYER + N_DISK);
  if (phi < phimin_[layerdisk])
    phimin_[layerdisk] = phi;
  if (phi > phimax_[layerdisk])
    phimax_[layerdisk] = phi;
}

void DTC::addLink(double phimin, double phimax) {
  DTCLink link(phimin, phimax);
  links_.push_back(link);
}

int DTC::addStub(std::pair<Stub*, L1TStub*> stub) {
  double phi = reco::reduceRange(stub.second->phi());
  bool overlaplayer = ((stub.second->layer() + 1) % 2 == 0);
  int added = 0;
  for (auto& link : links_) {
    if (link.inRange(phi, overlaplayer)) {
      added++;
      link.addStub(stub);
    }
  }
  return added;
}

void DTC::clean() {
  for (auto& link : links_) {
    link.clean();
  }
}
