#include "L1Trigger/TrackFindingTracklet/interface/VMRouterPhiCorrTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

using namespace std;
using namespace trklet;

VMRouterPhiCorrTable::VMRouterPhiCorrTable() : TETableBase(nullptr) { nbits_ = 14; }

void VMRouterPhiCorrTable::init(const Settings* settings, int layer, int bendbits, int rbits) {
  assert(bendbits == 3 || bendbits == 4);

  settings_ = settings;

  layer_ = layer;
  bendbits_ = bendbits;
  rbits_ = rbits;

  rbins_ = (1 << rbits);
  rmin_ = settings->rmean(layer - 1) - settings->drmax();
  rmax_ = settings->rmean(layer - 1) + settings->drmax();
  dr_ = 2 * settings->drmax() / rbins_;

  bendbins_ = (1 << bendbits);

  rmean_ = settings->rmean(layer - 1);

  for (int ibend = 0; ibend < bendbins_; ibend++) {
    for (int irbin = 0; irbin < rbins_; irbin++) {
      int value = getphiCorrValue(ibend, irbin);
      table_.push_back(value);
    }
  }

  if (settings->writeTable()) {
    writeVMTable("VMPhiCorrL" + std::to_string(layer_) + ".txt", false);
  }
}

int VMRouterPhiCorrTable::getphiCorrValue(int ibend, int irbin) const {
  double bend = trklet::benddecode(ibend, layer_ <= (int)N_PSLAYER);

  //for the rbin - calculate the distance to the nominal layer radius
  double Delta = (irbin + 0.5) * dr_ - settings_->drmax();

  //calculate the phi correction - this is a somewhat approximate formula
  double dphi = (Delta / 0.18) * (bend * settings_->stripPitch(false)) / rmean_;

  int idphi = 0;

  if (layer_ <= (int)N_PSLAYER) {
    idphi = dphi / settings_->kphi();
  } else {
    idphi = dphi / settings_->kphi1();
  }

  return idphi;
}

int VMRouterPhiCorrTable::lookupPhiCorr(int ibend, int rbin) {
  int index = ibend * rbins_ + rbin;
  assert(index < (int)table_.size());
  return table_[index];
}
