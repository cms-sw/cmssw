#ifndef gen_Pythia6PartonGun_h
#define gen_Pythia6PartonGun_h

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

  class Pythia6PartonGun : public Pythia6Gun {
  public:
    Pythia6PartonGun(const edm::ParameterSet&);
    ~Pythia6PartonGun() override;

  protected:
    void joinPartons(double qmax);

    // gun particle(s) characteristics
    //
    int fPartonID;
  };

}  // namespace gen

#endif
