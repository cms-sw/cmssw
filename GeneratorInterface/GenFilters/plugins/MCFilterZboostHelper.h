#ifndef MCFilterZboostHelper_h
#define MCFilterZboostHelper_h

#include "HepMC/SimpleVector.h"

namespace HepMC {
  class FourVector;
}

namespace MCFilterZboostHelper {

  HepMC::FourVector zboost(const HepMC::FourVector&, double);

}

#endif
