#include "DataFormats/CastorReco/interface/CastorJet.h"

using namespace reco;

CastorJet::CastorJet(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double
width, const double depth, const std::vector<CastorTower> usedTowers) {
  position_ = position;
  energy_ = energy;
  emEnergy_ = emEnergy;
  hadEnergy_ = hadEnergy;
  emtotRatio_ = emtotRatio;
  width_ = width;
  depth_ = depth;
  usedTowers_ = usedTowers;
}

CastorJet::~CastorJet() {

}
