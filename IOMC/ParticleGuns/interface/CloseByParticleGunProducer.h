#ifndef IOMC_ParticleGun_CloseByParticleGunProducer_H
#define IOMC_ParticleGun_CloseByParticleGunProducer_H

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {

  class CloseByParticleGunProducer : public BaseFlatGunProducer {
  public:
    CloseByParticleGunProducer(const ParameterSet&);
    ~CloseByParticleGunProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(Event& e, const EventSetup& es) override;

  protected:
    // data members
    bool fControlledByEta;
    double fEnMin, fEnMax, fEtaMin, fEtaMax, fRMin, fRMax, fZMin, fZMax, fDelta, fPhiMin, fPhiMax, fTMin, fTMax,
        fOffsetFirst;
    int fNParticles;
    bool fMaxEnSpread = false;
    bool fPointing = false;
    bool fOverlapping = false;
    bool fRandomShoot = false;
    bool fUseDeltaT = false;
    std::vector<int> fPartIDs;

    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_fieldToken;
  };
}  // namespace edm

#endif
