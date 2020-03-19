#ifndef FlatRandomEThetaGunProducer_H
#define FlatRandomEThetaGunProducer_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"

namespace edm {

  class FlatRandomEThetaGunProducer : public FlatBaseThetaGunProducer {
  public:
    FlatRandomEThetaGunProducer(const ParameterSet &);
    ~FlatRandomEThetaGunProducer() override;

  private:
    void produce(Event &e, const EventSetup &es) override;

  protected:
    // data members

    double fMinE;
    double fMaxE;
  };
}  // namespace edm

#endif
