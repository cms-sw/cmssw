#ifndef ExpoRandomPtGunSource_H
#define ExpoRandomPtGunSource_H

/** \class ExpoRandomPtGunSource
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"
#include "CLHEP/Random/RandExponential.h"
namespace edm
{
  
  class ExpoRandomPtGunSource : public BaseFlatGunSource
  {
  
  public:
    ExpoRandomPtGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~ExpoRandomPtGunSource();

  private:
   
    virtual bool produce(Event & e);
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;
    double            fMeanPt ;
    CLHEP::RandExponential * fRandomExpoGenerator;

  };
} 

#endif
