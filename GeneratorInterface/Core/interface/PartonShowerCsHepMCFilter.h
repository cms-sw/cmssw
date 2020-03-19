/**
 ** Description: Filter gen particles based on pdg_id and status code
 ** 
 ** @author bortigno
 ** @version 1.0 02.04.2015
*/

#ifndef PartonShowerCsHepMCFilter_h
#define PartonShowerCsHepMCFilter_h

#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PartonShowerCsHepMCFilter : public BaseHepMCFilter {
public:
  PartonShowerCsHepMCFilter(const edm::ParameterSet&);
  ~PartonShowerCsHepMCFilter() override;

  bool filter(const HepMC::GenEvent* evt) override;

private:
};

#endif
