#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

#include <iostream>
#include <iomanip>

using namespace reco;
using namespace std;


PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


ostream& reco::operator<<(ostream& out, 
			  const PFCandidate& c ) {
  
  if(!out) return out;
  
  out<<"\tPFCandidate type: "<<c.particleId()<<" pT=";
  out<<setiosflags(ios::right);
  out<<setiosflags(ios::fixed);
  out<<", pT="<<setw(7)<<c.pt();
  out<<", E ="<<setw(7)<<c.energy();
  
  out<<resetiosflags(ios::right|ios::fixed);

  out<< *(c.blockRef_)<<endl;
  
  return out;
}

