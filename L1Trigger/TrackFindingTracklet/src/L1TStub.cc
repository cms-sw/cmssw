#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"

using namespace std;
using namespace trklet;

L1TStub::L1TStub() {}

L1TStub::L1TStub(std::string DTClink,
                 int region,
                 int layerdisk,
                 std::string stubword,
                 int isPSmodule,
                 int isFlipped,
                 double x,
                 double y,
                 double z,
                 double bend,
                 double strip,
                 std::vector<int> tps) {
  DTClink_ = DTClink;
  layerdisk_ = layerdisk;
  region_ = region;
  stubword_ = stubword;
  eventid_ = -1;
  tps_ = tps;
  iphi_ = -1;
  iz_ = -1;
  layer_ = layerdisk;
  if (layerdisk >= N_LAYER) {
    layer_ = 1000 + layerdisk - N_LAYER + 1;
    if (z < 0.0)
      layer_ += 1000;
  }

  ladder_ = -1;
  module_ = -1;
  strip_ = strip;
  x_ = x;
  y_ = y;
  z_ = z;
  sigmax_ = -1.0;
  sigmaz_ = -1.0;
  pt_ = -1.0;
  bend_ = bend;
  isPSmodule_ = isPSmodule;
  isFlipped_ = isFlipped;

  allstubindex_ = 999;
}

void L1TStub::write(ofstream& out) {
  out << "Stub: " << DTClink_ << "\t" << region_ << "\t" << layerdisk_ << "\t" << stubword_ << "\t" << isPSmodule_
      << "\t" << isFlipped_ << "\t" << x_ << "\t" << y_ << "\t" << z_ << "\t" << bend_ << "\t" << strip_ << "\t"
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
    if (tp == std::abs(itp))
      return true;
  }

  return false;
}

bool L1TStub::tpmatch2(int tp) const {
  bool match1 = false;
  bool match2 = false;
  for (int itp : tps_) {
    if (tp == itp) {
      match1 = true;
    }
    if (tp == -itp) {
      match2 = true;
    }
  }

  return match1 && match2;
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
