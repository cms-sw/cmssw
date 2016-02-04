#ifndef FlatRandomPtThetaGunSource_H
#define FlatRandomPtThetaGunSource_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunSource.h"

namespace edm {
  
  class FlatRandomPtThetaGunSource : public FlatBaseThetaGunSource {
  
  public:
    FlatRandomPtThetaGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomPtThetaGunSource();

  private:
   
    virtual bool produce(Event & e);
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;

  };
} 

#endif
