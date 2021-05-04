#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

L1TStub::L1TStub() {}

L1TStub::L1TStub(int eventid,
                 vector<int> tps,
                 int iphi,
                 int iz,
                 int layer,
                 int ladder,
                 int module,
                 int strip,
                 double x,
                 double y,
                 double z,
                 double sigmax,
                 double sigmaz,
                 double pt,
                 double bend,
                 int isPSmodule,
                 int isFlipped) {
  eventid_ = eventid;
  tps_ = tps;
  iphi_ = iphi;
  iz_ = iz;
  layer_ = layer;
  ladder_ = ladder;
  module_ = module;
  strip_ = strip;
  x_ = x;
  y_ = y;
  z_ = z;
  sigmax_ = sigmax;
  sigmaz_ = sigmaz;
  pt_ = pt;
  bend_ = bend;
  isPSmodule_ = isPSmodule;
  isFlipped_ = isFlipped;

  allstubindex_ = 999;
}

void L1TStub::write(ofstream& out) {
  out << "Stub: " << layer_ + 1 << "\t" << ladder_ << "\t" << module_ << "\t" << strip_ << "\t" << eventid_ << "\t"
      << pt_ << "\t" << x_ << "\t" << y_ << "\t" << z_ << "\t" << bend_ << "\t" << isPSmodule_ << "\t" << isFlipped_
      << "\t" << tps_.size() << " \t";
  for (int itp : tps_) {
    out << itp << " \t";
  }
  out << endl;
}

void L1TStub::write(ostream& out) {
  out << "Stub: " << layer_ + 1 << "\t" << ladder_ << "\t" << module_ << "\t" << strip_ << "\t" << eventid_ << "\t"
      << pt_ << "\t" << x_ << "\t" << y_ << "\t" << z_ << "\t" << bend_ << "\t" << isPSmodule_ << "\t" << isFlipped_
      << "\t" << tps_.size() << " \t";
  for (int itp : tps_) {
    out << itp << " \t";
  }
  out << endl;
}

bool L1TStub::operator==(const L1TStub& other) const {
  return (other.iphi() == iphi_ && other.iz() == iz_ && other.layer() == layer_ && other.ladder() == ladder_ &&
          other.module() == module_);
}

void L1TStub::lorentzcor(double shift) {
  double r = this->r();
  double phi = this->phi() - shift / r;
  this->x_ = r * cos(phi);
  this->y_ = r * sin(phi);
}

double L1TStub::alpha(double pitch) const {
  if (isPSmodule())
    return 0.0;
  int flip = 1;
  if (isFlipped())
    flip = -1;
  if (z_ > 0.0) {
    return ((int)strip_ - 509.5) * pitch * flip / r2();
  }
  return -((int)strip_ - 509.5) * pitch * flip / r2();
}

double L1TStub::alphanorm() const {
  if (isPSmodule())
    return 0.0;
  int flip = 1;
  if (isFlipped())
    flip = -1;
  if (z_ > 0.0) {
    return ((int)strip_ - 509.5) * flip / 510.0;
  }
  return -((int)strip_ - 509.5) * flip / 510.0;
}

void L1TStub::setXY(double x, double y) {
  x_ = x;
  y_ = y;
}

bool L1TStub::tpmatch(int tp) const {
  for (int itp : tps_) {
    if (tp == itp)
      return true;
  }

  return false;
}

bool L1TStub::isTilted() const {
  //here layer_ runs 0-5 for barrel, >1000 for disk
  //disk modules and outer barrel modules are not tilted by construction
  if (layer_ >= N_PSLAYER)
    return false;

  assert(layer_ < N_PSLAYER);  // safety for acccessing # modules/plank
  if ((module_ <= N_TILTED_RINGS) || (module_ >= N_TILTED_RINGS + N_MOD_PLANK.at(layer_)))
    return true;
  return false;
}
