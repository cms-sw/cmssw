#include "DataFormats/ParticleFlowReco/interface/PFParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;


PFParticle::PFParticle() :
  PFTrack(),
  pdgCode_(0), 
  id_(0), 
  daughter1Id_(0), 
  daughter2Id_(0) 
{}


PFParticle::PFParticle(double charge, 
		       int pdgCode, 
		       unsigned id, 
		       unsigned daughter1Id, 
		       unsigned daughter2Id) : 
  PFTrack(charge),
  pdgCode_(pdgCode), 
  id_(id), 
  daughter1Id_(daughter1Id), 
  daughter2Id_(daughter2Id)   
{}
  

PFParticle::PFParticle(const PFParticle& other) :
  PFTrack(other), 
  pdgCode_(other.pdgCode_), 
  id_(other.id_), 
  daughter1Id_(other.daughter1Id_), 
  daughter2Id_(other.daughter2Id_)     
{}

std::ostream& reco::operator<<(std::ostream& out, 
			       const PFParticle& particle) {  
  if (!out) return out;  

  const reco::PFTrajectoryPoint& closestApproach = 
    particle.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out << "Particle charge = " << particle.charge() 
      << ", pdgcode = " << particle.pdgCode()
      << ", Pt = " << closestApproach.momentum().Pt() 
      << ", P = " << closestApproach.momentum().P() << std::endl
      << "\tR0 = " << closestApproach.positionXYZ().Rho()
      <<" Z0 = " << closestApproach.positionXYZ().Z() << std::endl
      << "\tnumber of trackers crossed = " 
      << particle.nTrajectoryMeasurements() << std::endl;

  return out;
}
