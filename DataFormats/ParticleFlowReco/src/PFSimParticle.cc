#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;


PFSimParticle::PFSimParticle() :
  PFTrack(),
  pdgCode_(0), 
  id_(0),
  motherId_(0)
{}


PFSimParticle::PFSimParticle(double charge, int pdgCode,
		       unsigned id, int motherId,
		       const vector<int>& daughterIds) : 
  PFTrack(charge),
  pdgCode_(pdgCode), 
  id_(id), 
  motherId_(motherId), 
  daughterIds_(daughterIds)   
{}
  

PFSimParticle::PFSimParticle(const PFSimParticle& other) :
  PFTrack(other), 
  pdgCode_(other.pdgCode_), 
  id_(other.id_), 
  motherId_(other.motherId_), 
  daughterIds_(other.daughterIds_)
{}

ostream& reco::operator<<(ostream& out, 
			  const PFSimParticle& particle) {  
  if (!out) return out;  

  const reco::PFTrajectoryPoint& closestApproach = 
    particle.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out<<"Particle #"<<particle.id()
     <<", mother = "<<particle.motherId()
     <<", charge = "<<particle.charge() 
     <<", pdgcode = "<<particle.pdgCode()
     <<", Pt = "<<closestApproach.momentum().Pt() 
     <<", P = "<<closestApproach.momentum().P();

  out<<"\tdaughters : ";
  for(unsigned i=0; i<particle.daughterIds_.size(); i++) 
    out<<particle.daughterIds_[i]<<" ";
  
  out<<endl;
  for(unsigned i=0; i<particle.trajectoryPoints_.size(); i++) 
    out<<particle.trajectoryPoints_[i]<<endl;

  return out;
}
