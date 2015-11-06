#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/JetInputHepMC.h"


JetInputHepMC::JetInputHepMC() :
  ptMin(0.0)
{}


JetInputHepMC::~JetInputHepMC()
{}
                                                        

void JetInputHepMC::setIgnoredParticles(const std::vector<unsigned int> &particleIDs)
{
  ignoreParticleIDs = particleIDs;
  std::sort(ignoreParticleIDs.begin(), ignoreParticleIDs.end());
}


static inline bool isContained(const std::vector<unsigned int> &list, int id)
{
  unsigned int absId = (unsigned int)(id > 0 ? id : -id);
  std::vector<unsigned int>::const_iterator pos =
                        std::lower_bound(list.begin(), list.end(), absId);
  return pos != list.end() && *pos == absId;
}


bool JetInputHepMC::isIgnored(int pdgId) const
{
  return isContained(ignoreParticleIDs, pdgId);
}


JetInputHepMC::ParticleVector JetInputHepMC::operator () (
                                const HepMC::GenEvent* event) const
{
  ParticleVector particles;
  for (HepMC::GenEvent::particle_const_iterator iter = event->particles_begin();
       iter != event->particles_end(); ++iter)
    particles.push_back(*iter);
                         
  std::sort(particles.begin(), particles.end());
  unsigned int size = particles.size();

  ParticleBitmap selected(size, false);
  ParticleBitmap invalid(size, false);

  for(unsigned int i = 0; i < size; i++) {
    const HepMC::GenParticle *particle = particles[i];
    if (invalid[i]) continue;
    if (particle->status() == 1) selected[i] = true;
  }

  ParticleVector result; 
  for(unsigned int i = 0; i < size; i++) {
    const HepMC::GenParticle *particle = particles[i];
    if (!selected[i] || invalid[i]) continue;

    //if (isIgnored(particle->pdg_id())) continue;
                         
    if (particle->momentum().perp() >= ptMin) result.push_back(particle);

  }
                                     
  return result;
}
