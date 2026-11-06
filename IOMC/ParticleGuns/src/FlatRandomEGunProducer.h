#ifndef IOMC_ParticleGuns_FlatRandomEGunProducer_h
#define IOMC_ParticleGuns_FlatRandomEGunProducer_h

#include "BaseFlatGunProducer.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;

  class FlatRandomEGunProducer : public BaseFlatGunProducer {
  public:
    FlatRandomEGunProducer(const ParameterSet& pset);
    ~FlatRandomEGunProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    void produce(Event& e, const EventSetup& es) override;

  protected:
    double fMinE;
    double fMaxE;
  };
}

#endif
