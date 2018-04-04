#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TPFUtils.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

std::pair<float,float> l1tpf::propagateToCalo(const math::XYZTLorentzVector& iMom, const math::XYZTLorentzVector& iVtx, double iCharge, double iBField) {
    BaseParticlePropagator particle = BaseParticlePropagator(RawParticle(iMom,iVtx),0.,0.,iBField);
    particle.setCharge(iCharge);
    particle.propagateToEcalEntrance(false);
    double ecalShowerDepth = reco::PFCluster::getDepthCorrection(particle.momentum().E(),false,false);
    math::XYZVector point = math::XYZVector(particle.vertex())+math::XYZTLorentzVector(particle.momentum()).Vect().Unit()*ecalShowerDepth;
    return std::make_pair(point.eta(), point.phi());
}

