#ifndef FlatRandomPtGunProducer_H
#define FlatRandomPtGunProducer_H

/** \class FlatRandomPtGunProducer
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 12/2005 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{
  
  class FlatRandomPtGunProducer : public BaseFlatGunProducer
  {
  
  public:
    FlatRandomPtGunProducer(const ParameterSet & pset);
    virtual ~FlatRandomPtGunProducer();
   
    virtual void produce(Event & e, const EventSetup& es) override;

  private:
    
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;

  };
} 

#endif
