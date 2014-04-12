#ifndef FlatRandomOneOverPtGunProducer_H
#define FlatRandomOneOverPtGunProducer_H

/** \class FlatRandomOneOverPtGunProducer
 *
 * Generates single particle gun flat in (1/pt) in HepMC format
 **************************************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{
  
  class FlatRandomOneOverPtGunProducer : public BaseFlatGunProducer
  {
  
  public:
    FlatRandomOneOverPtGunProducer(const ParameterSet & pset);
    virtual ~FlatRandomOneOverPtGunProducer();
   
    virtual void produce(Event & e, const EventSetup& es) override;

  private:
    
    // data members
    
    double            fMinOneOverPt   ;
    double            fMaxOneOverPt   ;

  };
} 

#endif
