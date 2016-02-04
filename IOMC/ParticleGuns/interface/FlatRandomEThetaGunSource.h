#ifndef FlatRandomEThetaGunSource_H
#define FlatRandomEThetaGunSource_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunSource.h"

namespace edm {

  class FlatRandomEThetaGunSource : public FlatBaseThetaGunSource {
  
  public:
    FlatRandomEThetaGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomEThetaGunSource();

  private:
   
    virtual bool produce(Event &e);
    
  protected :
  
    // data members
    
    double            fMinE   ;
    double            fMaxE   ;

  };
} 

#endif
