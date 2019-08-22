#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6PartonGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6PartonEGun : public Pythia6PartonGun {
  public:
    Pythia6PartonEGun(const edm::ParameterSet&);
    ~Pythia6PartonEGun() override;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    double fMinEta;
    double fMaxEta;
    double fMinE;
    double fMaxE;
  };

}  // namespace gen

#endif
