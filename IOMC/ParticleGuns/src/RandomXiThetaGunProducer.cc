/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *
 ****************************************************************************/

#include "IOMC/ParticleGuns/interface/RandomXiThetaGunProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

RandomXiThetaGunProducer::RandomXiThetaGunProducer(const edm::ParameterSet& pset) : 
  verbosity(pset.getUntrackedParameter<unsigned int>("verbosity", 0)),

  particleId(pset.getParameter<unsigned int>("particleId")),

  energy(pset.getParameter<double>("energy")),
  xi_min(pset.getParameter<double>("xi_min")),
  xi_max(pset.getParameter<double>("xi_max")),
  theta_x_mean(pset.getParameter<double>("theta_x_mean")),
  theta_x_sigma(pset.getParameter<double>("theta_x_sigma")),
  theta_y_mean(pset.getParameter<double>("theta_y_mean")),
  theta_y_sigma(pset.getParameter<double>("theta_y_sigma")),

  nParticlesSector45(pset.getParameter<unsigned int>("nParticlesSector45")),
  nParticlesSector56(pset.getParameter<unsigned int>("nParticlesSector56"))
{
  produces<HepMCProduct>("unsmeared");
}

//----------------------------------------------------------------------------------------------------

void RandomXiThetaGunProducer::produce(edm::Event &e, const edm::EventSetup& es) 
{
  if (verbosity)
    printf("===================== %u:%llu =====================\n", e.id().run(), e.id().event());

  // get conditions
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  ESHandle<HepPDT::ParticleDataTable> pdgTable;
  es.getData(pdgTable);

  // prepare HepMC event
  HepMC::GenEvent *fEvt = new HepMC::GenEvent();
  fEvt->set_event_number(e.id().event());
   
  HepMC::GenVertex *vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0., 0.));
  fEvt->add_vertex(vtx);

  const HepPDT::ParticleData *pData = pdgTable->particle(HepPDT::ParticleID(particleId));
  double mass = pData->mass().value();

  // generate particles
  unsigned int barcode = 0;

  for (unsigned int i = 0; i < nParticlesSector45; i++)
  {
    barcode++;
    GenerateParticle(+1., mass, barcode, engine, vtx);
  }

  for (unsigned int i = 0; i < nParticlesSector56; i++)
  {
    barcode++;
    GenerateParticle(-1., mass, barcode, engine, vtx);
  }

  // save output
  std::unique_ptr<HepMCProduct> output(new HepMCProduct()) ;
  output->addHepMCData(fEvt);
  e.put(std::move(output), "unsmeared");
}

//----------------------------------------------------------------------------------------------------

void RandomXiThetaGunProducer::GenerateParticle(double z_sign, double mass, unsigned int barcode,
  CLHEP::HepRandomEngine* engine, HepMC::GenVertex *vtx) const
{
  const double xi = CLHEP::RandFlat::shoot(engine, xi_min, xi_max);
  const double theta_x = CLHEP::RandGauss::shoot(engine, theta_x_mean, theta_x_sigma);
  const double theta_y = CLHEP::RandGauss::shoot(engine, theta_y_mean, theta_y_sigma);

  if (verbosity)
    printf("xi = %.4f, theta_x = %E, theta_y = %E, z_sign = %.0f\n", xi, theta_x, theta_y, z_sign);

  const double cos_theta = sqrt(1. - theta_x*theta_x - theta_y*theta_y);

  const double p_nom = sqrt(energy*energy - mass*mass);
  const double p = p_nom * (1. - xi);
  const double e = sqrt(p*p + mass*mass);

  HepMC::FourVector momentum(
    p * theta_x,
    p * theta_y,
    z_sign * p * cos_theta,
    e
  );

  HepMC::GenParticle* particle = new HepMC::GenParticle(momentum, particleId, 1);
  particle->suggest_barcode(barcode);
  vtx->add_particle_out(particle);
}

//----------------------------------------------------------------------------------------------------

RandomXiThetaGunProducer::~RandomXiThetaGunProducer()
{
}
