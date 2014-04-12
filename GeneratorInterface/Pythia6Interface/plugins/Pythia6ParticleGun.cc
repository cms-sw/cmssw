
#include <iostream>

#include "Pythia6ParticleGun.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace gen;

Pythia6ParticleGun::Pythia6ParticleGun( const ParameterSet& pset ) :
   Pythia6Gun(pset)
{
   
   ParameterSet pgun_params = 
      pset.getParameter<ParameterSet>("PGunParameters"); 
   fPartIDs    = pgun_params.getParameter< std::vector<int> >("ParticleID");

}

Pythia6ParticleGun::~Pythia6ParticleGun()
{
}

