#include "DataFormats/EgammaReco/interface/EcalCluster.h"

using namespace reco;

EcalCluster::EcalCluster(const float& energy, const math::XYZPoint& position) {
  position_ = position;
  energy_ = energy;
}

EcalCluster::~EcalCluster() {

}
