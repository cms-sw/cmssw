#ifndef Sherpa_Source_h
#define Sherpa_Source_h

/** \class SherpaSource
 *
 * 
 *   Martin Niegel
 ***************************************/

#include "SHERPA-MC/Sherpa.H"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"
#include "CLHEP/Random/RandFlat.h"
          




namespace CLHEP
{
  class RandFlat ;
  class HepRandomEngine;
}

namespace edm
{
  class SherpaSource : public GeneratedInputSource {
  public:

    /// Constructor
    SherpaSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~SherpaSource();


  private:
    
    virtual bool produce(Event & e);
    void clear();
    
    HepMC::GenEvent  *evt;
    std::string libDir_,resultDir_;

  
    SHERPA::Sherpa Generator;
   
      
    HepRandomEngine* fRandomEngine;
    CLHEP::RandFlat*        fRandomGenerator; 


  };
} 

#endif
