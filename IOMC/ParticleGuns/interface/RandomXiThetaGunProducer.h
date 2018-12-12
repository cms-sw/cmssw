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

namespace CLHEP { class HepRandomEngine; }
namespace edm {
  
class RandomXiThetaGunProducer : public one::EDProducer<>
{
  public:
    RandomXiThetaGunProducer(const ParameterSet&);
    ~RandomXiThetaGunProducer() override = default;

  private:
    virtual void produce(Event&, const EventSetup&) override;
    void generateParticle(double z_sign, double mass, unsigned int barcode, HepMC::GenVertex *vtx) const;

    unsigned int verbosity_;
    unsigned int particleId_;

    double energy_;
    double xi_min_, xi_max_;
    double theta_x_mean_, theta_x_sigma_;
    double theta_y_mean_, theta_y_sigma_;

    unsigned int nParticlesSector45_;
    unsigned int nParticlesSector56_;

    CLHEP::HepRandomEngine* engine_;
};

} 

#endif
