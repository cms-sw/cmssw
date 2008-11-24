#include "DataFormats/CastorReco/interface/CastorEgamma.h"
#include "FWCore/Framework/interface/Event.h"

using namespace reco;
using namespace edm;

CastorEgamma::CastorEgamma(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy, const double emtotRatio, const double
width, const double depth, const std::vector<CastorJet> usedJets) {
  position_ = position;
  energy_ = energy;
  emEnergy_ = emEnergy;
  hadEnergy_ = hadEnergy;
  emtotRatio_ = emtotRatio;
  width_ = width;
  depth_ = depth;
  usedJets_ = usedJets;
}

CastorEgamma::~CastorEgamma() {

}
