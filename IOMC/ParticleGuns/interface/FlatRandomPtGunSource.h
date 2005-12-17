#ifndef FlatRandomPtGunSource_H
#define FlatRandomPtGunSource_H

/** \class FlatRandomPtGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 12/2005 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"
#include "FWCore/Framework/interface/BranchDescription.h"

namespace edm
{
  
  class FlatRandomPtGunSource : public BaseFlatGunSource
  {
  
  public:
    FlatRandomPtGunSource(const ParameterSet &, const InputSourceDescription&  );
    virtual ~FlatRandomPtGunSource();

  private:
   
    virtual std::auto_ptr<EventPrincipal> read();
    
  protected :
  
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;

    // service stuff needed to put "product" into edm::Event
    BranchDescription fBranchDesc ;
    	    	
  };
} 

#endif
