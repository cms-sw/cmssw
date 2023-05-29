
#include <iostream>

#include "Pythia6PartonGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace gen;

Pythia6PartonGun::Pythia6PartonGun(const ParameterSet& pset) : Pythia6Gun(pset) {
  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");
  fPartonID = pgun_params.getParameter<int>("PartonID");
}

Pythia6PartonGun::~Pythia6PartonGun() {}

void Pythia6PartonGun::joinPartons(double qmax) {
  int njoin = 2;
  int ijoin[] = {1, 2};
  pyjoin_(njoin, ijoin);
  int i1 = 1;
  int i2 = 2;
  pyshow_(i1, i2, qmax);

  return;
}
