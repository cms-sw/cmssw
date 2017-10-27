#ifndef RandomtXiGunProducer_H
#define RandomtXiGunProducer_H

#include "IOMC/ParticleGuns/interface/BaseRandomtXiGunProducer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"


namespace edm {
  
  class RandomtXiGunProducer : public BaseRandomtXiGunProducer {
  
  public:
    RandomtXiGunProducer(const ParameterSet &);
    ~RandomtXiGunProducer() override;

  private:
   
    void produce(Event & e, const EventSetup& es) override;

    HepMC::FourVector make_particle(double t,double Xi,double phi,int PartID, int direction);
    double Minimum_t(double xi) {double partE = fpEnergy*(1.-xi);
                                 double massSQ= pow(PData->mass().value(),2);
                                 double partP = sqrt(partE*partE-massSQ);
                                 return -2.*(sqrt(fpEnergy*fpEnergy-massSQ)*partP-fpEnergy*partE+massSQ);
                                };
    
  protected :
  
    // data members
    
    double            fMint   ;
    double            fMaxt   ;
    double            fMinXi  ;
    double            fMaxXi  ;

  };
} 
#endif
