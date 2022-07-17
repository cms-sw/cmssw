#ifndef IOMC_ParticleGun_CloseByParticleGunProducer_H
#define IOMC_ParticleGun_CloseByParticleGunProducer_H

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm {

  class CloseByParticleGunProducer : public BaseFlatGunProducer {
  public:
    CloseByParticleGunProducer(const ParameterSet&);
    ~CloseByParticleGunProducer() override;

  private:
    void produce(Event& e, const EventSetup& es) override;

  protected:
    // data members
    bool fControlledByEta;
    double fEnMin, fEnMax, fEtaMin, fEtaMax, fRMin, fRMax, fZMin, fZMax, fDelta, fPhiMin, fPhiMax;
    int fNParticles;
    bool fMaxEnSpread = false;
    bool fPointing = false;
    bool fOverlapping = false;
    bool fRandomShoot = false;
    std::vector<int> fPartIDs;
  };
}  // namespace edm

#endif
