#include "DataFormats/CastorReco/interface/CastorCell.h"

using namespace reco;

CastorCell::CastorCell(const double energy, const ROOT::Math::XYZPoint& position) {
  position_ = position;
  energy_ = energy;
}

CastorCell::~CastorCell() {

}
