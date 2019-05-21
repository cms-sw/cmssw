#ifndef IOMC_ParticleGuns_RandomMultiParticlePGunProducer_H
#define IOMC_ParticleGuns_RandomMultiParticlePGunProducer_H

#include <vector>
#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm {

  class RandomMultiParticlePGunProducer : public BaseFlatGunProducer {
  public:
    RandomMultiParticlePGunProducer(const ParameterSet& pset);
    ~RandomMultiParticlePGunProducer() override {}

    void produce(Event& e, const EventSetup& es) override;

  private:
    // data members
    std::vector<double> fProbParticle_;
    std::vector<double> fProbP_;
    double fMinP_;
    double fMaxP_;
    int fBinsP_;
    double fDeltaP_;
  };
}  // namespace edm
#endif
