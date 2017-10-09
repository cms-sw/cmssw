#ifndef ExpoRandomPGunProducer_H
#define ExpoRandomPGunProducer_H

/** \class ExpoRandomPGunProducer
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 * modificed by S.Abdulin 04/02/2011 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{

  class ExpoRandomPGunProducer : public BaseFlatGunProducer
  {

  public:
    ExpoRandomPGunProducer(const ParameterSet & pset);
    virtual ~ExpoRandomPGunProducer();

  private:

    virtual void produce(Event & e, const EventSetup& es);

  protected :

    // data members

    double            fMinP   ;
    double            fMaxP   ;

  };
}

#endif
