#ifndef gen_Pythia6PtYDistGun_h
#define gen_Pythia6PtYDistGun_h

#include "Pythia6ParticleGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class PtYDistributor;

  class Pythia6PtYDistGun : public Pythia6ParticleGun {
  public:
    Pythia6PtYDistGun(const edm::ParameterSet&);
    ~Pythia6PtYDistGun() override;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    PtYDistributor* fPtYGenerator;
  };
}  // namespace gen
#endif
