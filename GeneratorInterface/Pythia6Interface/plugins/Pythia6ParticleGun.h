#ifndef gen_Pythia6ParticleGun_h
#define gen_Pythia6ParticleGun_h

/** \class Pythia6Gun
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 02/2009 
 ***************************************/

#include <string>
#include <vector>

#include "Pythia6Gun.h"

namespace gen {

  // class Pythia6Service;

  class Pythia6ParticleGun : public Pythia6Gun {
  public:
    Pythia6ParticleGun(const edm::ParameterSet&);
    ~Pythia6ParticleGun() override;

  protected:
    // gun particle(s) characteristics
    //
    std::vector<int> fPartIDs;
  };

}  // namespace gen

#endif
