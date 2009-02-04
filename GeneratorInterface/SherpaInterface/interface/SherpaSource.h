#ifndef Sherpa_Source_h
#define Sherpa_Source_h

/** \class SherpaSource
 *
 * 
 *   Martin Niegel niegel@cern.ch
 ***************************************/

#include "SHERPA-MC/Sherpa.H"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"
#include "CLHEP/Random/RandFlat.h"
          

namespace edm
{
  class SherpaSource : public GeneratedInputSource {
  public:

    SherpaSource(const ParameterSet &, const InputSourceDescription &);
    virtual ~SherpaSource();

  private:
    
    virtual bool produce(Event & e);
    void clear();
    
    HepMC::GenEvent  *evt;
    std::string libDir_,resultDir_;
    SHERPA::Sherpa Generator;
   
  };
} 


#include "SHERPA-MC/Random.H"
class CMS_RNG: public ATOOLS::External_RNG {
  public: 
    double Get();
};


#endif
