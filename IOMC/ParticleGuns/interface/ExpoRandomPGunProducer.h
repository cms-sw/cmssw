#ifndef ExpoRandomPGunProducer_H
#define ExpoRandomPGunProducer_H

/** \class ExpoRandomPGunProducer
 *
 * Generates single particle gun in HepMC format
 * Jean-Roch Vlimant
 * modificed by S.Abdulin 04/02/2011 
 ***************************************/

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

namespace edm {

  class ExpoRandomPGunProducer : public BaseFlatGunProducer {
  public:
    ExpoRandomPGunProducer(const ParameterSet& pset);
    ~ExpoRandomPGunProducer() override;

  private:
    void produce(Event& e, const EventSetup& es) override;

  protected:
    // data members

    double fMinP;
    double fMaxP;
  };
}  // namespace edm

#endif
