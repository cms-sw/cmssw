#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TPFUtils.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

std::pair<float, float> l1tpf::propagateToCalo(const math::XYZTLorentzVector& iMom,
                                               const math::XYZTLorentzVector& iVtx,
                                               double iCharge,
                                               double iBField) {
  BaseParticlePropagator prop = BaseParticlePropagator(RawParticle(iMom, iVtx, iCharge), 0., 0., iBField);
  prop.propagateToEcalEntrance(false);
  double ecalShowerDepth = reco::PFCluster::getDepthCorrection(prop.particle().momentum().E(), false, false);
  math::XYZVector point = math::XYZVector(prop.particle().vertex()) +
                          math::XYZTLorentzVector(prop.particle().momentum()).Vect().Unit() * ecalShowerDepth;
  return std::make_pair(point.eta(), point.phi());
}
