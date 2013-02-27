#ifndef FileRandomKEThetaGunProducer_H
#define FileRandomKEThetaGunProducer_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"
#include <vector>

namespace edm {

  class FileRandomKEThetaGunProducer : public FlatBaseThetaGunProducer {
  
  public:
    FileRandomKEThetaGunProducer(const ParameterSet &);
    virtual ~FileRandomKEThetaGunProducer();

  private:
   
    virtual void produce(Event &e, const EventSetup& es) override;
    
  protected :
  
    // data members
    
    std::vector<double> kineticE, fdistn;
    int                 particleN;
  };
} 

#endif
