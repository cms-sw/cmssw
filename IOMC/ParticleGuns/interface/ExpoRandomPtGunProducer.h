#ifndef ExpoRandomPtGunProducer_H
#define ExpoRandomPtGunProducer_H

/** \class ExpoRandomPtGunProducer
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{
  
  class ExpoRandomPtGunProducer : public BaseFlatGunProducer
  {
  
  public:
    ExpoRandomPtGunProducer(const ParameterSet & pset);
    ~ExpoRandomPtGunProducer() override;

  private:
   
    void produce(Event & e, const EventSetup& es) override;
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;
    double            fMeanPt ;
  };
} 

#endif
