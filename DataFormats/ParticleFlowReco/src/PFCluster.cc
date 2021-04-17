#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "Math/GenVector/etaMax.h"

using namespace std;
using namespace reco;

const math::XYZPoint PFCluster::dummyVtx_(0, 0, 0);

PFCluster::PFCluster(PFLayer::Layer layer, double energy, double x, double y, double z)
    : CaloCluster(energy, math::XYZPoint(x, y, z), PFLayer::toCaloID(layer), CaloCluster::particleFlow),
      posrep_(position_.Rho(), position_.Eta(), position_.Phi()),
      time_(-99.),
      depth_(0.),
      layer_(layer) {}

void PFCluster::reset() {
  energy_ = 0;
  position_ *= 0;
  posrep_ *= 0;
  time_ = -99.;
  layer_ = PFLayer::NONE;
  rechits_.clear();

  CaloCluster::reset();
}

void PFCluster::resetHitsAndFractions() {
  rechits_.clear();
  hitsAndFractions_.clear();
}

void PFCluster::addRecHitFraction(const reco::PFRecHitFraction& frac) {
  rechits_.push_back(frac);

  addHitAndFraction(frac.recHitRef()->detId(), frac.fraction());
}

double PFCluster::getDepthCorrection(double energy, bool isBelowPS, bool isHadron) {
  double corrA = depthCorA_;
  double corrB = depthCorB_;
  if (isBelowPS) {
    corrA = depthCorAp_;
    corrB = depthCorBp_;
  }
  return isHadron ? corrA : corrA * (corrB + log(energy));
}

void PFCluster::setLayer(PFLayer::Layer layer) {
  // cout<<"calling PFCluster::setLayer "<<layer<<endl;
  layer_ = layer;
  caloID_ = PFLayer::toCaloID(layer);
  // cout<<"done "<<caloID_<<endl;
}

PFLayer::Layer PFCluster::layer() const {
  // cout<<"calling PFCluster::layer "<<caloID()<<" "<<PFLayer::fromCaloID( caloID() )<<endl;
  if (layer_ != PFLayer::NONE)
    return layer_;
  return PFLayer::fromCaloID(caloID());
}

PFCluster& PFCluster::operator=(const PFCluster& other) {
  CaloCluster::operator=(other);
  rechits_ = other.rechits_;
  energy_ = other.energy_;
  position_ = other.position_;
  posrep_ = other.posrep_;

  return *this;
}

std::ostream& reco::operator<<(std::ostream& out, const PFCluster& cluster) {
  if (!out)
    return out;

  const math::XYZPoint& pos = cluster.position();
  const PFCluster::REPPoint& posrep = cluster.positionREP();
  const std::vector<reco::PFRecHitFraction>& fracs = cluster.recHitFractions();

  out << "PFcluster "
      << ", layer: " << cluster.layer() << "\tE = " << cluster.energy() << "\tXYZ: " << pos.X() << "," << pos.Y() << ","
      << pos.Z() << " | "
      << "\tREP: " << posrep.Rho() << "," << posrep.Eta() << "," << posrep.Phi() << " | " << fracs.size() << " rechits";

  for (unsigned i = 0; i < fracs.size(); i++) {
    // PFRecHit is not available, print the detID
    if (!fracs[i].recHitRef().isAvailable())
      out << cluster.printHitAndFraction(i) << ", ";
    else
      out << fracs[i] << ", ";
  }

  return out;
}
