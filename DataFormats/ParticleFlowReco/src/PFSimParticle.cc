#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

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
                             const vector<int>& daughterIds,
			     unsigned rectrackId, 
			     const std::vector<unsigned>& recHitContrib, 
			     const std::vector<double>& recHitContribFrac ):
  PFTrack(charge),
  pdgCode_(pdgCode), 
  id_(id), 
  motherId_(motherId), 
  daughterIds_(daughterIds),
  rectrackId_(rectrackId), 
  recHitContrib_(recHitContrib), 
  recHitContribFrac_(recHitContribFrac)
{}


PFSimParticle::PFSimParticle(const PFSimParticle& other) :
  PFTrack(other), 
  pdgCode_(other.pdgCode_), 
  id_(other.id_), 
  motherId_(other.motherId_), 
  daughterIds_(other.daughterIds_),
  rectrackId_(other.rectrackId_),
  recHitContrib_(other.recHitContrib_),
  recHitContribFrac_(other.recHitContribFrac_)
{}

ostream& reco::operator<<(ostream& out, 
                          const PFSimParticle& particle) {  
  if (!out) return out;  

  const reco::PFTrajectoryPoint& closestApproach = 
    particle.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);
  
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  
  out<<"Particle #"<<particle.id()
     <<", mother = "<<setw(2)<<particle.motherId();
  
  out<<setprecision(1);
  out<<", charge = "<<setw(5)<<particle.charge();
  out<<setprecision(3);
  
  //   modif to get "name" of particle from pdg code
  int partId = particle.pdgCode();
  std::string name;
  
  // We have here a subset of particles only. 
  // To be filled according to the needs.
  switch(partId) {
	  case    1: { name = "d"; break; } 
	  case    2: { name = "u"; break; } 
	  case    3: { name = "s"; break; } 
	  case    4: { name = "c"; break; } 
	  case    5: { name = "b"; break; } 
	  case    6: { name = "t"; break; } 
	  case   -1: { name = "~d"; break; } 
	  case   -2: { name = "~u"; break; } 
	  case   -3: { name = "~s"; break; } 
	  case   -4: { name = "~c"; break; } 
	  case   -5: { name = "~b"; break; } 
	  case   -6: { name = "~t"; break; } 
	  case   11: { name = "e-"; break; }
	  case  -11: { name = "e+"; break; }
	  case   12: { name = "nu_e"; break; }
	  case  -12: { name = "~nu_e"; break; }
	  case   13: { name = "mu-"; break; }
	  case  -13: { name = "mu+"; break; }
	  case   14: { name = "nu_mu"; break; }
	  case  -14: { name = "~nu_mu"; break; }
	  case   15: { name = "tau-"; break; }
	  case  -15: { name = "tau+"; break; }
	  case   16: { name = "nu_tau"; break; }
	  case  -16: { name = "~nu_tau"; break; }
	  case   21: { name = "gluon"; break; }
	  case   22: { name = "gamma"; break; }
	  case   23: { name = "Z0"; break; }
	  case   24: { name = "W+"; break; }
	  case   25: { name = "H0"; break; }
	  case  -24: { name = "W-"; break; }
	  case  111: { name = "pi0"; break; }
	  case  113: { name = "rho0"; break; }
	  case  223: { name = "omega"; break; }
	  case  333: { name = "phi"; break; }
	  case  443: { name = "J/psi"; break; }
	  case  553: { name = "Upsilon"; break; }
	  case  130: { name = "K0L"; break; }
	  case  211: { name = "pi+"; break; }
	  case -211: { name = "pi-"; break; }
	  case  213: { name = "rho+"; break; }
	  case -213: { name = "rho-"; break; }
	  case  221: { name = "eta"; break; }
	  case  331: { name = "eta'"; break; }
	  case  441: { name = "etac"; break; }
	  case  551: { name = "etab"; break; }
	  case  310: { name = "K0S"; break; }
	  case  311: { name = "K0"; break; }
	  case -311: { name = "Kbar0"; break; }
	  case  321: { name = "K+"; break; }
	  case -321: { name = "K-"; break; }
	  case  411: { name = "D+"; break; }
	  case -411: { name = "D-"; break; }
	  case  421: { name = "D0"; break; }
	  case  431: { name = "Ds_+"; break; }
	  case -431: { name = "Ds_-"; break; }
	  case  511: { name = "B0"; break; }
	  case  521: { name = "B+"; break; }
	  case -521: { name = "B-"; break; }
	  case  531: { name = "Bs_0"; break; }
	  case  541: { name = "Bc_+"; break; }
	  case -541: { name = "Bc_+"; break; }
	  case  313: { name = "K*0"; break; }
	  case -313: { name = "K*bar0"; break; }
	  case  323: { name = "K*+"; break; }
	  case -323: { name = "K*-"; break; }
	  case  413: { name = "D*+"; break; }
	  case -413: { name = "D*-"; break; }
	  case  423: { name = "D*0"; break; }
	  case  513: { name = "B*0"; break; }
	  case  523: { name = "B*+"; break; }
	  case -523: { name = "B*-"; break; }
	  case  533: { name = "B*_s0"; break; }
	  case  543: { name = "B*_c+"; break; }
	  case -543: { name = "B*_c-"; break; }
	  case  1114: { name = "Delta-"; break; }
	  case -1114: { name = "Deltabar+"; break; }
	  case -2112: { name = "nbar0"; break; }
	  case  2112: { name = "n"; break; }
	  case  2114: { name = "Delta0"; break; }
	  case -2114: { name = "Deltabar0"; break; }
	  case  3122: { name = "Lambda0"; break; }
	  case -3122: { name = "Lambdabar0"; break; }
	  case  3112: { name = "Sigma-"; break; }
	  case -3112: { name = "Sigmabar+"; break; }
	  case  3212: { name = "Sigma0"; break; }
	  case -3212: { name = "Sigmabar0"; break; }
	  case  3214: { name = "Sigma*0"; break; }
	  case -3214: { name = "Sigma*bar0"; break; }
	  case  3222: { name = "Sigma+"; break; }
	  case -3222: { name = "Sigmabar-"; break; }
	  case  2212: { name = "p"; break; }
	  case -2212: { name = "~p"; break; }
	  case -2214: { name = "Delta-"; break; }
	  case  2214: { name = "Delta+"; break; }
	  case -2224: { name = "Deltabar--"; break; }
	  case  2224: { name = "Delta++"; break; }
	  default: { 
    name = "unknown"; 
    cout << "Unknown code : " << partId << endl;
  }   
  }
  
  out<<", pdg="<<setw(6)<<particle.pdgCode() << setw(6)<< name
    
    // end of modif to get name from pdg code
    
     <<", pT ="<<setw(7)<<closestApproach.momentum().Pt() 
     <<", E  ="<<setw(7)<<closestApproach.momentum().E();
  
  out<<resetiosflags(ios::right|ios::fixed);
  
  out<<"\tdaughters : ";
  for(unsigned i=0; i<particle.daughterIds_.size(); i++) 
    out<<particle.daughterIds_[i]<<" ";
  
  //   out<<endl;
  //   for(unsigned i=0; i<particle.trajectoryPoints_.size(); i++) 
  //     out<<particle.trajectoryPoints_[i]<<endl;
  
  return out;
}
