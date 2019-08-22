#ifndef BaseHepMCFilter_H
#define BaseHepMCFilter_H

// J.Bendavid

// base class for simple filter to run inside of HadronizerFilter for
// multiple hadronization attempts on lhe events

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class BaseHepMCFilter {
public:
  BaseHepMCFilter();
  virtual ~BaseHepMCFilter();

  virtual bool filter(const HepMC::GenEvent* evt) = 0;
};

#endif
