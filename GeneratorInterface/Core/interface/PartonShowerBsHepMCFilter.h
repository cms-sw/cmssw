/**
 ** Description: Filter gen particles based on pdg_id and status code
 ** 
 ** @author bortigno
 ** @version 1.0 02.04.2015
*/


#ifndef __PARTONSHOWERBSHEPMCFILTER__
#define __PARTONSHOWERBSHEPMCFILTER__


#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PartonShowerBsHepMCFilter : public BaseHepMCFilter{

public:

  PartonShowerBsHepMCFilter( const edm::ParameterSet & );
  ~PartonShowerBsHepMCFilter();

  virtual bool filter(const HepMC::GenEvent* evt);

private:

};


#endif