#ifndef FlatRandomPtThetaGunProducer_H
#define FlatRandomPtThetaGunProducer_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"

namespace edm {
  
  class FlatRandomPtThetaGunProducer : public FlatBaseThetaGunProducer {
  
  public:
    FlatRandomPtThetaGunProducer(const ParameterSet &);
    ~FlatRandomPtThetaGunProducer() override;

  private:
   
    void produce(Event & e, const EventSetup& es) override;
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;

  };
} 

#endif
