#ifndef IOMC_ParticleGuns_FlatRandomMultiParticlePGunProducer_H
#define IOMC_ParticleGuns_FlatRandomMultiParticlePGunProducer_H

#include <vector>
#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm {

  class FlatRandomMultiParticlePGunProducer : public BaseFlatGunProducer {
  
  public:
    FlatRandomMultiParticlePGunProducer(const ParameterSet & pset);
    ~FlatRandomMultiParticlePGunProducer() override;
    
    void produce(Event &e, const EventSetup& es) override;

  private:
    
    // data members
    std::vector<double> fProbParticle_;
    double              fMinP_;
    double              fMaxP_;
    
  };
} 
#endif
