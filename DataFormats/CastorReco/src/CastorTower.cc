#include "DataFormats/CastorReco/interface/CastorTower.h"

using namespace reco;

CastorTower::CastorTower(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double
width, const double depth, const std::vector<CastorCell> usedCells) {
  position_ = position;
  energy_ = energy;
  emEnergy_ = emEnergy;
  hadEnergy_ = hadEnergy;
  emtotRatio_ = emtotRatio;
  width_ = width;
  depth_ = depth;
  usedCells_ = usedCells;
}

CastorTower::~CastorTower() {

}
