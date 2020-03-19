#ifndef GaussRandomPThetaGunProducer_H
#define GaussRandomPThetaGunProducer_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"

namespace edm {

  class GaussRandomPThetaGunProducer : public FlatBaseThetaGunProducer {
  public:
    GaussRandomPThetaGunProducer(const ParameterSet &);
    ~GaussRandomPThetaGunProducer() override;

  private:
    void produce(Event &e, const EventSetup &es) override;

  protected:
    // data members

    double fMeanP;
    double fSigmaP;
    double fMeanTheta;
    double fSigmaTheta;
  };
}  // namespace edm

#endif
