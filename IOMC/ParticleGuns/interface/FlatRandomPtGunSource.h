#ifndef FlatRandomPtGunSource_H
#define FlatRandomPtGunSource_H

/** \class FlatRandomPtGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 12/2005 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

namespace edm
{
  
  class FlatRandomPtGunSource : public BaseFlatGunSource
  {
  
  public:
    FlatRandomPtGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomPtGunSource();

  private:
   
    virtual bool produce(Event & e);
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;

  };
} 

#endif
