#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

thread_local ParticleTable* ParticleTable::myself = nullptr;


double 
ParticleTable::PDGmass(int pdgID) const {
  auto info = theTable()->particle(HepPDT::ParticleID(pdgID));
  if(info) {
    return info->mass().value();
  }
  return kInvalidMass;
}

double 
ParticleTable::PDGcTau(int pdgID) const {
  auto info = theTable()->particle(HepPDT::ParticleID(pdgID));
  double ct = kInvalidCtau;
  if (info) {
    // The lifetime is 0. in the Pythia Particle Data Table !
    //    ct=tab->theTable()->particle(ParticleID(myId))->lifetime().value();

    // Get it from the width (apparently Gamma/c!)
    double w = info->totalWidth().value();
    if ( w != 0. && pdgID != 1000022 ) { 
      ct = 6.582119e-25 / w / 10.;   // ctau in cm 
    } else {
      // Temporary fix of a bug in the particle data table
      unsigned amyId = abs(pdgID);
      if ( amyId != 22 &&    // photon 
	   amyId != 11 &&    // e+/-
	   amyId != 10 &&    // nu_e
	   amyId != 12 &&    // nu_mu
	   amyId != 14 &&    // nu_tau
	   amyId != 1000022 && // Neutralino
	   amyId != 1000039 && // Gravitino
	   amyId != 2112 &&  // neutron/anti-neutron
	   amyId != 2212 &&  // proton/anti-proton
	   amyId != 101 &&   // Deutreron etc..
	   amyId != 102 &&   // Deutreron etc..
	   amyId != 103 &&   // Deutreron etc..
	   amyId != 104 ) {  // Deutreron etc.. 
	ct = 0.;
	/* */
      }
    }
  }

  /*
  std::cout << setw(20) << setprecision(18) 
       << "myId/ctau/width = " << myId << " " 
       << ct << " " << w << endl;  
  */
  return ct;
}

