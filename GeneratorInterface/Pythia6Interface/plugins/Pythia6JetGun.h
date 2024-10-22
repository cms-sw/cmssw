#ifndef gen_Pythia6JetGun_h
#define gen_Pythia6JetGun_h

#include "Pythia6ParticleGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6JetGun : public Pythia6ParticleGun {
  public:
    Pythia6JetGun(const edm::ParameterSet&);
    ~Pythia6JetGun() override;
    // void produce( edm::Event&, const edm::EventSetup& ) ;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    double fMinEta;
    double fMaxEta;
    double fMinE;
    double fMaxE;
    double fMinP;
    double fMaxP;
  };

}  // namespace gen

#endif
