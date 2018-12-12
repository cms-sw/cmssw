/**
 ** Description: Filter gen particles based on pdg_id and status code
 ** 
 ** @author bortigno
 ** @version 1.0 02.04.2015
*/


#ifndef __PARTONSHOWERCSHEPMCFILTER__
#define __PARTONSHOWERCSHEPMCFILTER__


#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PartonShowerCsHepMCFilter : public BaseHepMCFilter{

public:

  PartonShowerCsHepMCFilter( const edm::ParameterSet & );
  ~PartonShowerCsHepMCFilter();

  virtual bool filter(const HepMC::GenEvent* evt);

private:

};


#endif
