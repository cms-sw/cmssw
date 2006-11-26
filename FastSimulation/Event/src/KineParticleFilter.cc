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

  // Change eta cuts to cos**2(theta) cuts (less CPU consuming)
  double cosMax = (exp(2.*etaMax)-1.) / (exp(2.*etaMax)+1.);
  cos2Max = cosMax*cosMax;

  double etaPreshMin = 1.479;
  double etaPreshMax = 1.594;
  double cosPreshMin = (exp(2.*etaPreshMin)-1.) / (exp(2.*etaPreshMin)+1.);
  double cosPreshMax = (exp(2.*etaPreshMax)-1.) / (exp(2.*etaPreshMax)+1.);
  cos2PreshMin = cosPreshMin*cosPreshMin;
  cos2PreshMax = cosPreshMax*cosPreshMax;

}

bool KineParticleFilter::isOKForMe(const RawParticle* p) const
{

  // Do not consider quarks, gluons, Z, W, strings, diquarks
  // ... and supesymmetric particles
  int pId = abs(p->pid());

  // Vertices are coming with pId = 0
  if ( pId != 0 ) { 
    bool particleCut = ( pId > 10  && pId != 12 && pId != 14 && 
			 pId != 16 && pId != 18 && pId != 21 &&
			 (pId < 23 || pId > 40  ) &&
			 (pId < 81 || pId > 100 ) && pId != 2101 &&
			 pId != 3101 && pId != 3201 && pId != 1103 &&
			 pId != 2103 && pId != 2203 && pId != 3103 &&
			 pId != 3203 && pId != 3303 );
    //    particleCut = particleCut || pId == 0;

    if ( !particleCut ) return false;


  //  bool kineCut = pId == 0;
  // Cut on kinematic properties
    // Cut on the energy of all particles
    bool eneCut = p->e() >= EMin;
    if (!eneCut) return false;

    // Cut on the transverse momentum of charged particles
    bool pTCut = p->charge()==0 || p->perp()>=pTMin;
    if (!pTCut) return false;

    // Cut on eta if the origin vertex is close to the beam
    //    bool etaCut = (p->vertex()-mainVertex).perp()>5. || fabs(p->eta())<=etaMax;
    bool etaCut = (p->vertex()-mainVertex).perp()>5. || p->vect().cos2Theta()<= cos2Max;

    /*
    if ( etaCut != etaCut2 ) 
      cout << "WANRNING ! etaCut != etaCut2 " 
	   << etaCut << " " 
	   << etaCut2 << " "
	   << (p->eta()) << " " << etaMax << " " 
	   << p->vect().cos2Theta() << " " << cos2Max << endl; 
    */
    if (!etaCut) return false;

  }
  // The Kinematic cuts (for particles only, not for vertices)
  //  bool kineCut = ( etaCut && eneCut && pTCut ) || pId==0;

  // Cut on the origin vertex position (prior to the ECAL for all 
  // particles, except for muons  ! Just modified: Muons included as well !
  HepLorentzVector position = p->vertex();
  double radius = position.perp();
  double zed = fabs(position.z());
  double cos2Tet = position.vect().cos2Theta();
  // Ecal entrance
  bool ecalAcc = ( (radius<129.01 && zed<317.01) ||
		   (cos2Tet>cos2PreshMin && cos2Tet<cos2PreshMax 
		    && radius<171.11 && zed<317.01) );

  return ecalAcc;

  // OBSOLETE
  // Hcal entrance
  //  bool hcalAcc = (radius<285. && zed<560.);
  // The vertex position condition
  // This condition is actually not valid: pions may start showering in the 
  // ECAL, and then no decay to muons should be allowed.
  //  bool vertexCut = (hcalAcc && pId == 13 ) || ecalAcc;
  //  if ( !vertexCut ) return false;
  //  return true;

}
