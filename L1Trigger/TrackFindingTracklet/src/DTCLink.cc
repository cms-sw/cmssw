#include "L1Trigger/TrackFindingTracklet/interface/DTCLink.h"

using namespace std;
using namespace trklet;

DTCLink::DTCLink(double phimin, double phimax) {
  if (phimin > M_PI) {
    phimin -= 2 * M_PI;
    phimax -= 2 * M_PI;
  }
  assert(phimax > phimin);
  phimin_ = phimin;
  phimax_ = phimax;
}

void DTCLink::addStub(std::pair<Stub*, L1TStub*> stub) { stubs_.push_back(stub); }

bool DTCLink::inRange(double phi, bool overlaplayer) {
  double phimax = phimax_;
  double phimin = phimin_;
  if (overlaplayer) {
    double dphi = phimax - phimin;
    assert(dphi > 0.0);
    assert(dphi < M_PI);
    phimin -= dphi / 6.0;
    phimax += dphi / 6.0;
  }
  return (phi < phimax && phi > phimin) || (phi + 2 * M_PI < phimax && phi + 2 * M_PI > phimin);
}
