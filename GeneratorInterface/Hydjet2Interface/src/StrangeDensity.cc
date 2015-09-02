
#include "GeneratorInterface/Hydjet2Interface/interface/StrangeDensity.h"

NAStrangeDensity::NAStrangeDensity() {
  fTemperature = 0.*GeV;
  fBaryonPotential = 0.*GeV;
  fStrangePotential = 0.*GeV;
  fNMax = 5;
}
// compute hadron system strangeness density
double NAStrangeDensity::StrangenessDensity(DatabasePDG* database) {
  double meanStrangenessDensity = 0.;
  for(int particleIndex = 0; particleIndex < database->GetNParticles(); particleIndex++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(particleIndex);
    double particleDensity = ParticleNumberDensity(particle);
    meanStrangenessDensity += particleDensity*particle->GetStrangeness();
  }
  return meanStrangenessDensity;
}

// compute hadron number density
double NAStrangeDensity::ParticleNumberDensity(ParticlePDG* pDef) {
  double particleMass = pDef->GetMass();
  int particleStrangeness = int(pDef->GetStrangeness());
  double particleBaryon = pDef->GetBaryonNumber();
  //compute chemical potential
  double particleChemPotential = fBaryonPotential*particleBaryon + 
    fStrangePotential*particleStrangeness;
  //compute degeneracy factor
  double particleDegFactor = 2*pDef->GetSpin() + 1.;     // IA: In ParticlePDG() GetSpin() returns spin not 2*spin !!
  double d = 1.;//for fermions
  if(int(2*pDef->GetSpin())%2 == 0)//it gives 0 for Spin = 0,2,4,.. and it gives 1 for Spin = 1,3,7,
    d = -1; //for bosons

  double prefactor;
  double postfactor;
  prefactor = (particleDegFactor*particleMass*particleMass*
	       fTemperature/hbarc/hbarc/hbarc)/(2.*N_PI*N_PI);  
  postfactor = 0.;
 
  for(int n = 1; n <= fNMax; n++) {
    postfactor += pow(-d,n+1)/(n)*exp(n*particleChemPotential/fTemperature)*
      HankelKn(2,n*particleMass/fTemperature);
  }
  return prefactor*postfactor;
}

