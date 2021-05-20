#include "L1Trigger/TrackFindingTracklet/interface/VMRouterPhiCorrTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include <filesystem>

using namespace std;
using namespace trklet;

VMRouterPhiCorrTable::VMRouterPhiCorrTable(Settings const& settings) : TETableBase(settings) { nbits_ = 14; }

void VMRouterPhiCorrTable::init(int layer, int bendbits, int rbits) {
  assert(bendbits == 3 || bendbits == 4);

  layer_ = layer;
  bendbits_ = bendbits;
  rbits_ = rbits;

  rbins_ = (1 << rbits);
  rmin_ = settings_.rmean(layer - 1) - settings_.drmax();
  rmax_ = settings_.rmean(layer - 1) + settings_.drmax();
  dr_ = 2 * settings_.drmax() / rbins_;

  bendbins_ = (1 << bendbits);

  rmean_ = settings_.rmean(layer - 1);

  for (int ibend = 0; ibend < bendbins_; ibend++) {
    for (int irbin = 0; irbin < rbins_; irbin++) {
      int value = getphiCorrValue(ibend, irbin);
      table_.push_back(value);
    }
  }

  if (settings_.writeTable()) {
    writeVMTable(settings_.tablePath(), "VMPhiCorrL" + std::to_string(layer_) + ".tab", false);
  }
}

int VMRouterPhiCorrTable::getphiCorrValue(int ibend, int irbin) const {
  assert(layer_ > 0.0 && layer_ <= (int)N_LAYER);
  double bend = -settings_.benddecode(ibend, layer_ - 1, layer_ <= (int)N_PSLAYER);

  //for the rbin - calculate the distance to the nominal layer radius
  double Delta = (irbin + 0.5) * dr_ - settings_.drmax();

  //calculate the phi correction - this is a somewhat approximate formula
  double dphi = (Delta / 0.18) * (bend * settings_.stripPitch(layer_ <= (int)N_PSLAYER)) / rmean_;

  int idphi = 0;

  if (layer_ <= (int)N_PSLAYER) {
    idphi = dphi / settings_.kphi();
  } else {
    idphi = dphi / settings_.kphi1();
  }

  return idphi;
}

int VMRouterPhiCorrTable::lookupPhiCorr(int ibend, int rbin) {
  int index = ibend * rbins_ + rbin;
  assert(index < (int)table_.size());
  return table_[index];
}
