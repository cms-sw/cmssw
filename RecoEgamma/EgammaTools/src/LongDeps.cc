#include "RecoEgamma/EgammaTools/interface/LongDeps.h"

using namespace hgcal;

LongDeps::LongDeps(float radius,
                   const std::vector<float>& energyPerLayer,
                   float energyEE,
                   float energyFH,
                   float energyBH,
                   const std::set<int>& layers)
    : energyPerLayer_(energyPerLayer),
      radius_(radius),
      energyEE_(energyEE),
      energyFH_(energyFH),
      energyBH_(energyBH),
      layers_(layers) {
  lay_Efrac10_ = 0;
  lay_Efrac90_ = 0;
  float lay_energy = 0;
  float e4 = 0.;
  // NB: energyPerLayer_ is 1-indexed
  for (unsigned lay = 1; lay < energyPerLayer_.size(); ++lay) {
    lay_energy += energyPerLayer_[lay];
    if (lay < 5)
      e4 += energyPerLayer_[lay];
    if (lay_Efrac10_ == 0 && lay_energy > 0.1 * energyEE_) {
      lay_Efrac10_ = lay;
    }
    if (lay_Efrac90_ == 0 && lay_energy > 0.9 * energyEE_) {
      lay_Efrac90_ = lay;
    }
  }
  float etot = energyEE_ + energyFH_ + energyBH_;
  e4oEtot_ = (etot > 0.) ? e4 / etot : -1.;
}
