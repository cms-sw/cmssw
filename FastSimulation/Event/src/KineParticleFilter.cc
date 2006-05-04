#include "FastSimulation/Event/interface/KineParticleFilter.h"

using namespace std;

#include <iostream>


KineParticleFilter::KineParticleFilter(const edm::ParameterSet& kine) 
  : BaseRawParticleFilter() 
{

  // Set the kinematic cuts
  etaMax = kine.getParameter<double>("etaMax"); // Upper abs(eta) bound
  pTMin  = kine.getParameter<double>("pTMin");  // Lower pT  bound (charged, in GeV/c)
  EMin   = kine.getParameter<double>("EMin");   // Lower E  bound (all, in GeV)

}

bool KineParticleFilter::isOKForMe(const RawParticle* p) const
{

  // Do not consider quarks, gluons, Z, W, strings, diquarks
  // ... and supesymmetric particles
  int pId = abs(p->pid());

  bool particleCut = ( pId > 10  && pId != 12 && pId != 14 && 
		       pId != 16 && pId != 18 && pId != 21 &&
		       (pId < 23 || pId > 40  ) &&
		       (pId < 81 || pId > 100 ) && pId != 2101 &&
		       pId != 3101 && pId != 3201 && pId != 1103 &&
		       pId != 2103 && pId != 2203 && pId != 3103 &&
		       pId != 3203 && pId != 3303 );
  // Vertices are coming with pId = 0
  particleCut = particleCut || pId == 0;


  // Cut on the origin vertex position (prior to the ECAL for all 
  // particles, except for muons
  HepLorentzVector position = p->vertex();
  double radius = position.perp();
  double zed = fabs(position.z());
  double eta = fabs(position.eta());
  // Ecal entrance
  bool ecalAcc = ( (radius<129.01 && zed<317.01) ||
		(eta>1.479 && eta<1.594 && radius<171.11 && zed<317.01) );
  // Hcal entrance
  bool hcalAcc = (radius<285. && zed<560.);
  // The vertex position condition
  bool vertexCut = (hcalAcc && pId == 13) || ecalAcc;

  // Cut on kinematic properties
  // Cut on eta if the origin vertex is close to the beam
  bool etaCut = (p->vertex()-mainVertex).perp()>5. || fabs(p->eta())<=etaMax;
  if ( !etaCut ) return false;
  // Cut on the energy of all particles
  bool eneCut = p->e() >= EMin;
  // Cut on the transverse momentum of charged particles
  bool pTCut = p->PDGcharge()==0 || p->perp()>=pTMin;
  // The Kinematic cuts (for particles only, not for vertices)
  bool kineCut = ( etaCut && eneCut && pTCut ) || pId==0;

  return ( particleCut && vertexCut && kineCut );

}
