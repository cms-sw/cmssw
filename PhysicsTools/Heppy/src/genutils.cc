#include "PhysicsTools/Heppy/interface/genutils.h"

int heppy::GenParticleRefHelper::motherKey(const reco::GenParticle &gp, int index) { return gp.motherRef(index).key(); }
int heppy::GenParticleRefHelper::daughterKey(const reco::GenParticle &gp, int index) { return gp.daughterRef(index).key(); }
