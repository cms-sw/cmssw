/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#ifndef RandomXiThetaGunProducer_H
#define RandomXiThetaGunProducer_H

#include "FWCore/Framework/interface/one/EDProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

namespace edm {
  
class RandomXiThetaGunProducer : public one::EDProducer<>
{

  public:
    RandomXiThetaGunProducer(const ParameterSet &);

    virtual ~RandomXiThetaGunProducer();

  private:
    virtual void produce(Event & e, const EventSetup& es) override;
    
    void GenerateParticle(double z_sign, double mass, unsigned int barcode, CLHEP::HepRandomEngine* engine, HepMC::GenVertex *vtx) const;

    unsigned int verbosity;

    unsigned int particleId;

    double energy;
    double xi_min, xi_max;
    double theta_x_mean, theta_x_sigma;
    double theta_y_mean, theta_y_sigma;

    unsigned int nParticlesSector45;
    unsigned int nParticlesSector56;
};

} 

#endif
