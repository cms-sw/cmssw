#include "FastSimulation/Particle/interface/KineParticleFilter.h"

using namespace std;

#include <iostream>


KineParticleFilter::KineParticleFilter() {
  // Set the kinematic cuts
  vector<double> defaultCuts;
  defaultCuts.push_back(-5.0);  // Lower eta bound
  defaultCuts.push_back(+5.0);  // Upper eta bound
  defaultCuts.push_back(0.);    // Lower phi bound (in degrees)
  defaultCuts.push_back(360.);  // Upper phi bound (in degrees)
  defaultCuts.push_back(0.20);  // Lower pT  bound (charged, in GeV/c)
  defaultCuts.push_back(9999.); // Upper pT  bound (charged, in GeV/c)
  defaultCuts.push_back(0.10);  // Lower E  bound (all, in GeV)
  defaultCuts.push_back(9999.); // Upper E  bound (all, in GeV)

  //  ConfigurableVector<double> theCuts(defaultCuts,"Kine:Cuts");

  //  if ( theCuts.size() == 8 ) {
  //    etaMin = theCuts[0];
  //    etaMax = theCuts[1];
  //    phiMin = theCuts[2];
  //    phiMax = theCuts[3];
  //    pTMin  = theCuts[4];
  //    pTMax  = theCuts[5];
  //    EMin   = theCuts[6];
  //    EMax   = theCuts[7];
  //  } else {
    etaMin = defaultCuts[0];
    etaMax = defaultCuts[1];
    phiMin = defaultCuts[2];
    phiMax = defaultCuts[3];
    pTMin  = defaultCuts[4];
    pTMax  = defaultCuts[5];
    EMin   = defaultCuts[6];
    EMax   = defaultCuts[7];
    //  }

}

bool KineParticleFilter::accept(int pId, const HepLorentzVector& p, 
				         const HepLorentzVector& v ) const
{
  // input in cm, converted to mm in the RawParticle
  RawParticle part(p,v*10.);
  part.setID(pId);

  return BaseRawParticleFilter::accept(&part);
}

bool KineParticleFilter::accept(int pId, const HepLorentzVector& v ) const 
{

  // input in cm, converted to mm in the RawParticle
  double value = 0.5* ( max(pTMin,EMin) + min(pTMax,EMax) );
  RawParticle part(HepLorentzVector(value,0.,0.,value),v*10.);
  part.setID(pId);

  return BaseRawParticleFilter::accept(&part);
}

bool KineParticleFilter::accept(int pId) const 
{
  double value = 0.5* ( max(pTMin,EMin) + min(pTMax,EMax) );
  RawParticle part(HepLorentzVector(value,0.,0.,value));
  part.setID(pId);

  return BaseRawParticleFilter::accept(&part);
}

bool KineParticleFilter::isOKForMe(const RawParticle* p) const
{

  // Do not consider quarks, gluons, Z, W, strings, diquarks
  // ... and supesymmetric particles
  int pId = abs(p->pid());

  bool particle = ( pId > 10  && pId != 12 && pId != 14 && 
                    pId != 16 && pId != 18 && pId != 21 &&
		   (pId < 23 || pId > 40  ) &&
                   (pId < 81 || pId > 100 ) && pId != 2101 &&
                    pId != 3101 && pId != 3201 && pId != 1103 &&
                    pId != 2103 && pId != 2203 && pId != 3103 &&
                    pId != 3203 && pId != 3303 );

  HepLorentzVector position = p->vertex();
  double eta = fabs(position.eta());
  bool ecal = (
       (position.perp() < 1290.1 && fabs(position.z()) < 3170.1) ||
       (eta > 1.479 && eta < 1.594 &&
	position.perp() < 1711.1 && fabs(position.z()) < 3170.1) );
  bool hcal = (position.perp() < 2850. && fabs(position.z()) < 5600.);
    
  /*
  cout << " pID " << pId << " " << particle 
       << " ecal " << (pId == 13 || ecal)
       << " PhiMin " << (p->phi()+M_PI)*180./M_PI << " " 
                     << ( (p->phi()+M_PI)*180./M_PI  >= phiMin ) 
                     << " " << phiMin << " " 
       << " PhiMax " << ( (p->phi()+M_PI)*180./M_PI  <= phiMax )
                     << " " << phiMax << " " 
       << " EtaMin " << p->eta() << " " 
                     << ( (p->vertex()-mainVertex).perp() > 50. ||
			  p->eta()   >= etaMin )
                     << " " << etaMin << " " 
       << " EtaMax " << ( (p->vertex()-mainVertex).perp() > 50. ||
			  p->eta()  <= etaMax )
                     << " " << etaMax << " " 
       << " PtMin " << p->perp() << " " 
                    << ( p->PDGcharge() == 0 || p->perp() >= pTMin )
                    << " " << pTMin << " " 
       << " PtMax " << ( p->PDGcharge() == 0 || p->perp() <= pTMax )
                    << " " << pTMax << " " 
       << " EMin " << p->e() << " " 
                   << ( p->e() >= EMin )
                   << " " << EMin << " " 
       << " EMax " << ( p->e() <= EMax ) 
                   << " " << EMax << " " << endl;
*/
  return ( particle &&
// cut all particles but muons beyond ECAL entrance
  	  ( (hcal && pId == 13) || ecal) && 
  	  (p->phi()+M_PI)*180./M_PI  >= phiMin && // phi lower cut
  	  (p->phi()+M_PI)*180./M_PI  <= phiMax && // phi upper cut
	  ((p->vertex()-mainVertex).perp() > 50. ||          // eta cut if vertex is
  	  (p->eta()   >= etaMin && p->eta()  <= etaMax )) && // close to IP
	  (p->PDGcharge() == 0 ||
	  (p->perp() >= pTMin  && p->perp() <= pTMax)) && // pT cut for charged
	   p->e() >= EMin && p->e() <= EMax ); // Energy cut for all

  }
