/*
  Ludmila Malinina  malinina@lav01.sinp.msu.ru,   SINP MSU/Moscow and JINR/Dubna
  Ionut Arsene  i.c.arsene@fys.uio.no,            Oslo University and ISS-Bucharest
  Date        : 2007/05/30
*/

#include "TVector3.h"
#include "GeneratorInterface/Hydjet2Interface/interface/InitialState.h"
#include "GeneratorInterface/Hydjet2Interface/interface/HadronDecayer.h"
#include <iostream> 
#include <fstream>

void InitialState::Evolve(List_t &secondaries, ParticleAllocator &allocator, double weakDecayLimit) {
  // Particle indexes are set for primaries already from InitialStateHydjet::Initialize()
   
  // particle list iterators
  LPIT_t it;
  LPIT_t e;

  // Decay loop
  // Note that the decay products are always added at the end of list so the unstable products are
  // decayed when the iterator reaches them (used for cascade decays)
  for(it = secondaries.begin(), e = secondaries.end(); it != e; ++it) {

    // if the decay procedure was applied already skip ... (e.g. particles from pythia history information)
    if(it->GetDecayed()) {
      continue;
    }

    // generate the decay time; if particle is stable or set to stable decayTime=0
    double decayTime = GetDecayTime(*it, weakDecayLimit);
    it->SetLastInterTime(it->T() + decayTime);
    TVector3 shift(it->Mom().BoostVector());
    shift *= decayTime; 
    it->SetDecayed();

    // if decayTime>0 then apply the decay procedure (only 2 or 3 body decays)
    if(decayTime > 0.) {

      it->Pos(it->Pos() += TLorentzVector(shift, 0.));
      it->T(it->T() + decayTime);
      Decay(secondaries, *it, allocator, fDatabase);
    }
    // if particle is stable just continue
  }
}
