#include "FastSimulation/Event/interface/KineParticleFilter.h"

using namespace std;

#include <iostream>


KineParticleFilter::KineParticleFilter() : BaseRawParticleFilter() {
  // Set the kinematic cuts
  vector<double> defaultCuts;
  defaultCuts.push_back(5.0);   // Upper abs(eta) bound
  defaultCuts.push_back(0.20);  // Lower pT  bound (charged, in GeV/c)
  defaultCuts.push_back(0.10);  // Lower E  bound (all, in GeV)

  //  ConfigurableVector<double> theCuts(defaultCuts,"Kine:Cuts");

  etaMax = defaultCuts[0];
  pTMin  = defaultCuts[1];
  EMin   = defaultCuts[2];

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
  bool ecalAcc = ( (radius<1290.1 && zed<3170.1) ||
		(eta>1.479 && eta<1.594 && radius<1711.1 && zed<3170.1) );
  // Hcal entrance
  bool hcalAcc = (radius<2850. && zed<5600.);
  // The vertex position condition
  bool vertexCut = (hcalAcc && pId == 13) || ecalAcc;

  // Cut on kinematic properties
  // Cut on eta if the origin vertex is close to the beam
  bool etaCut = (p->vertex()-mainVertex).perp()>50. || fabs(p->eta())<=etaMax;
  // Cut on the energy of all particles
  bool eneCut = p->e() >= EMin;
  // Cut on the transverse momentum of charged particles
  bool pTCut = p->PDGcharge()==0 || p->perp()>=pTMin;
  // The Kinematic cuts (for particles only, not for vertices)
  bool kineCut = ( etaCut && eneCut && pTCut ) || pId==0;

  return ( particleCut && vertexCut && kineCut );

}
