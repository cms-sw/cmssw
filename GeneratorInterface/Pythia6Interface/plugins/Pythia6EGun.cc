
#include <iostream>

#include "Pythia6EGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6EGun::Pythia6EGun(const ParameterSet& pset) : Pythia6ParticleGun(pset) {
  // ParameterSet defpset ;
  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");  // , defpset ) ;
  fMinEta = pgun_params.getParameter<double>("MinEta");                          // ,-2.2);
  fMaxEta = pgun_params.getParameter<double>("MaxEta");                          // , 2.2);
  fMinE = pgun_params.getParameter<double>("MinE");                              // ,  0.);
  fMaxE = pgun_params.getParameter<double>("MaxE");                              // ,  0.);
  fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");          //, false) ;
}

Pythia6EGun::~Pythia6EGun() {}

void Pythia6EGun::generateEvent(CLHEP::HepRandomEngine*) {
  Pythia6Service::InstanceWrapper guard(fPy6Service);  // grab Py6 instance

  // now actualy, start cooking up the event gun
  //

  // 1st, primary vertex
  //
  HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0.));

  // here re-create fEvt (memory)
  //
  fEvt = new HepMC::GenEvent();

  int ip = 1;
  for (size_t i = 0; i < fPartIDs.size(); i++) {
    int particleID = fPartIDs[i];  // this is PDG - need to convert to Py6 !!!
    int py6PID = HepPID::translatePDTtoPythia(particleID);
    int dum = 0;
    double ee = 0, the = 0, eta = 0;
    double mass = pymass_(particleID);

    // fill p(ip,5) (in PYJETS) with mass value right now,
    // because the (hardcoded) mstu(10)=1 will make py1ent
    // pick the mass from there
    pyjets.p[4][ip - 1] = mass;

    double phi = (fMaxPhi - fMinPhi) * pyr_(&dum) + fMinPhi;
    ee = (fMaxE - fMinE) * pyr_(&dum) + fMinE;
    eta = (fMaxEta - fMinEta) * pyr_(&dum) + fMinEta;
    the = 2. * atan(exp(-eta));

    py1ent_(ip, py6PID, ee, the, phi);

    double px = pyjets.p[0][ip - 1];  // pt*cos(phi) ;
    double py = pyjets.p[1][ip - 1];  // pt*sin(phi) ;
    double pz = pyjets.p[2][ip - 1];  // mom*cos(the) ;

    HepMC::FourVector p(px, py, pz, ee);
    HepMC::GenParticle* Part = new HepMC::GenParticle(p, particleID, 1);
    Part->suggest_barcode(ip);
    Vtx->add_particle_out(Part);

    if (fAddAntiParticle) {
      ip = ip + 1;
      HepMC::GenParticle* APart = addAntiParticle(ip, particleID, ee, eta, phi);
      if (APart)
        Vtx->add_particle_out(APart);
    }
    ip++;
  }

  fEvt->add_vertex(Vtx);

  // run pythia
  pyexec_();

  return;
}

DEFINE_FWK_MODULE(Pythia6EGun);
