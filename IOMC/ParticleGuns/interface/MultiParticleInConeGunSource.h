#ifndef MultiParticleInConeGunSource_H
#define MultiParticleInConeGunSource_H

/** \class MultiParticleInConeGunSource
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"
#include "CLHEP/Random/RandExponential.h"
namespace edm
{
  
  class MultiParticleInConeGunSource : public BaseFlatGunSource
  {
  
  public:
    MultiParticleInConeGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~MultiParticleInConeGunSource();

  private:
   
    virtual bool produce(Event & e);
    
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
