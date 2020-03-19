#ifndef gen_Pythia6PtGun_h
#define gen_Pythia6PtGun_h

#include "Pythia6PartonGun.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6PartonPtGun : public Pythia6PartonGun {
  public:
    Pythia6PartonPtGun(const edm::ParameterSet&);
    ~Pythia6PartonPtGun() override;

  protected:
    void generateEvent(CLHEP::HepRandomEngine*) override;

  private:
    double fMinEta;
    double fMaxEta;
    double fMinPt;
    double fMaxPt;
  };

}  // namespace gen

#endif
