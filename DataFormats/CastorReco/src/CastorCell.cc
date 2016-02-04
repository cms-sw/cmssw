#include "DataFormats/CastorReco/interface/CastorCell.h"

reco::CastorCell::CastorCell(const double energy, const ROOT::Math::XYZPoint& position) {
  position_ = position;
  energy_ = energy;
}

reco::CastorCell::~CastorCell() {

}
