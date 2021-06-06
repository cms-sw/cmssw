#ifndef IOMC_ParticleGuns_FileRandomMultiParticlePGunProducer_H
#define IOMC_ParticleGuns_FileRandomMultiParticlePGunProducer_H

#include <vector>
#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm {

  class FileRandomMultiParticlePGunProducer : public BaseFlatGunProducer {
  public:
    FileRandomMultiParticlePGunProducer(const ParameterSet& pset);
    ~FileRandomMultiParticlePGunProducer() override;

    void produce(Event& e, const EventSetup& es) override;

  private:
    // data members
    int fPBin_;
    int fEtaBin_;
    std::vector<double> fP_;
    std::map<int, std::vector<double> > fProbParticle_;
    double fEtaMin_;
    double fEtaBinWidth_;
    double fMinP_;
    double fMaxP_;
  };
}  // namespace edm
#endif
