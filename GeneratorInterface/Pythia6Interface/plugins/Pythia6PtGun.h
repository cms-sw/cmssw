#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6ParticleGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6PtGun : public Pythia6ParticleGun {
  public:
    Pythia6PtGun(const edm::ParameterSet&);
    ~Pythia6PtGun() override;
    // void produce( edm::Event&, const edm::EventSetup& ) ;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    double fMinEta;
    double fMaxEta;
    double fMinPt;
    double fMaxPt;
    bool fAddAntiParticle;
  };

}  // namespace gen

#endif
