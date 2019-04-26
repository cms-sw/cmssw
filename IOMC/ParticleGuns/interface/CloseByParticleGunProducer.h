#ifndef IOMC_ParticleGun_CloseByParticleGunProducer_H
#define IOMC_ParticleGun_CloseByParticleGunProducer_H

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm
{

  class CloseByParticleGunProducer : public BaseFlatGunProducer
  {

  public:
    CloseByParticleGunProducer(const ParameterSet &);
    ~CloseByParticleGunProducer() override;

  private:

    void produce(Event & e, const EventSetup& es) override;

  protected :

    // data members
    double fEnMin,fEnMax,fRMin,fRMax,fZMin,fZMax,fDelta,fPhiMin,fPhiMax;

    bool fPointing = false;
    bool fOverlapping = false;
    std::vector<int> fPartIDs;
  };
}

#endif
