#ifndef FlatRandomEGunSource_H
#define FlatRandomEGunSource_H

/** \class FlatRandomEGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 10/2005 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"
#include "FWCore/Framework/interface/BranchDescription.h"

namespace edm
{
  
  class FlatRandomEGunSource : public BaseFlatGunSource
  {
  
  public:
    FlatRandomEGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomEGunSource();

  private:
   
    virtual std::auto_ptr<EventPrincipal> read();
    
  protected :
  
    // data members
    
    double            fMinE   ;
    double            fMaxE   ;

    // service stuff needed to put "product" into edm::Event
    BranchDescription fBranchDesc ;
    	    	
  };
} 

#endif
