
#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <list>
#include <utility>
#include <iostream>

using std::list;
using std::pair;

MaterialEffectsSimulator:: MaterialEffectsSimulator(const RandomEngine* engine)
{ 
  random = engine;
  _theUpdatedState.clear(); 
}

MaterialEffectsSimulator::~MaterialEffectsSimulator() {
  // Don't delete the objects contained in the list
  _theUpdatedState.clear();
}

void MaterialEffectsSimulator::updateState(ParticlePropagator & Particle,
					 double radlen)

{
  // delete the pointeurs
  for( RHEP_const_iter deleteiter = beginDaughters(); 
       deleteiter!=endDaughters(); 
       ++deleteiter ) {
    delete (*deleteiter);
  }

  _theUpdatedState.clear();

  radLengths = radlen;
  if ( radLengths > 0. ) compute(Particle);

}
