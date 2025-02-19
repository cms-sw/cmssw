#ifndef FlatRandomEGunSource_H
#define FlatRandomEGunSource_H

/** \class FlatRandomEGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 10/2005 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

namespace edm
{

  class FlatRandomEGunSource : public BaseFlatGunSource
  {
  
  public:
    FlatRandomEGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomEGunSource();

  private:
   
    virtual bool produce(Event &e);
    
  protected :
  
    // data members
    
    double            fMinE   ;
    double            fMaxE   ;

  };
} 

#endif
