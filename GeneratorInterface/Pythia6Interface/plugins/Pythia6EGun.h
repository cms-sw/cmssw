#ifndef gen_Pythia6EGun_h
#define gen_Pythia6EGun_h

#include "Pythia6ParticleGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6EGun : public Pythia6ParticleGun {
  public:
    Pythia6EGun(const edm::ParameterSet&);
    ~Pythia6EGun() override;
    // void produce( edm::Event&, const edm::EventSetup& ) ;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    double fMinEta;
    double fMaxEta;
    double fMinE;
    double fMaxE;
    bool fAddAntiParticle;
  };

}  // namespace gen

#endif
