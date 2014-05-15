// JetInputHepMC declaration

#ifndef JetInputHepMC_h
#define JetInputHepMC_h

#include <vector>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>

#include <iostream>


class JetInputHepMC {
  public:
    typedef std::vector<bool>                       ParticleBitmap;
    typedef std::vector<const HepMC::GenParticle*>  ParticleVector;
    JetInputHepMC();
    ~JetInputHepMC();

    ParticleVector operator () (const HepMC::GenEvent *event) const;

    double getPtMin() const { return ptMin; }
    void setPtMin(double ptMin) { this->ptMin = ptMin; }

    void setIgnoredParticles(const std::vector<unsigned int> &particleIDs);

    bool isIgnored(int pdgId) const;

  private:
    std::vector<unsigned int>       ignoreParticleIDs;   
    double                          ptMin;
};

#endif // JetInputHepMC_h
