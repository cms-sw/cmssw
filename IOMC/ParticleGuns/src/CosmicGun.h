#ifndef CosmicGun_H
#define CosmicGun_H

/** \class CosmicGun
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 12/2005 
 ***************************************/

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{
  
  class CosmicGun : public BaseFlatGunProducer
  {
  
  public:
    CosmicGun(const ParameterSet & pset);
    virtual ~CosmicGun() override;
   
    virtual void produce(Event & e, const EventSetup& es) override;

  private:
    
    // data members
    
    double            fMinPt   ;
    double            fMaxPt   ;
    double           fMinEta ;
    double           fMaxEta ;

  };
  DEFINE_FWK_MODULE(CosmicGun) ;  
}
#endif
