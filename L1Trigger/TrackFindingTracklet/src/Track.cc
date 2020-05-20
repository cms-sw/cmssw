#include "L1Trigger/TrackFindingTracklet/interface/Track.h"

#include <algorithm>

using namespace std;
using namespace trklet;

Track::Track(TrackPars<int> ipars,
             int ichisqrphi,
             int ichisqrz,
             double chisqrphi,
             double chisqrz,
             int hitpattern,
             std::map<int, int> stubID,
             const std::vector<const L1TStub*>& l1stub,
             int seed) {
  ipars_ = ipars;
  ichisqrphi_ = ichisqrphi;
  ichisqrz_ = ichisqrz;

  chisqrphi_ = chisqrphi;
  chisqrz_ = chisqrz;

  hitpattern_ = hitpattern;

  nstubs_ = std::max((int)l1stub.size(), 6);

  stubID_ = stubID;
  l1stub_ = l1stub;

  seed_ = seed;
  duplicate_ = false;
  sector_ = -1;
}

double Track::phi0(const Settings* settings) const {
  double dphi = 2 * M_PI / N_SECTOR;
  double dphiHG = 0.5 * settings->dphisectorHG() - M_PI / N_SECTOR;
  double phimin = sector_ * dphi - dphiHG;
  double phimax = phimin + dphi + 2 * dphiHG;
  phimin -= M_PI / N_SECTOR;
  phimax -= M_PI / N_SECTOR;
  phimin = trklet::phiRange(phimin);
  phimax = trklet::phiRange(phimax);
  if (phimin > phimax)
    phimin -= 2 * M_PI;
  double phioffset = phimin;

  return ipars_.phi0() * settings->kphi0pars() + phioffset;
}
