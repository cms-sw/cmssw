#ifndef BaseHepMC3Filter_H
#define BaseHepMC3Filter_H

// base class for simple filter to run inside of HadronizerFilter for
// multiple hadronization attempts on lhe events

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"

//
// class declaration
//

class BaseHepMC3Filter {
public:
  BaseHepMC3Filter();
  ~BaseHepMC3Filter();

  bool filter(const HepMC3::GenEvent* evt) { return true; };
};

#endif
