#ifndef MultiParticleInConeGunProducer_H
#define MultiParticleInConeGunProducer_H

/** \class MultiParticleInConeGunProducer
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{
  
  class MultiParticleInConeGunProducer : public BaseFlatGunProducer
  {
  
  public:
    MultiParticleInConeGunProducer(const ParameterSet &);
    virtual ~MultiParticleInConeGunProducer();

  private:
   
    virtual void produce(Event & e, const EventSetup& es) override;
    
  protected :
  
    // data members
    double            fMinPt   ;
    double            fMaxPt   ;

    std::vector<int> fInConeIds;
    double fMinDeltaR;
    double fMaxDeltaR;
    double fMinMomRatio;
    double fMaxMomRatio;

    double fInConeMinEta;
    double fInConeMaxEta;
    double fInConeMinPhi;
    double fInConeMaxPhi;
    unsigned int fInConeMaxTry;

  };
} 

#endif
