/****************************************************************************
 * Authors:
 *   Jan Kašpar
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

RandomXiThetaGunProducer::RandomXiThetaGunProducer(const edm::ParameterSet& iConfig) :
  verbosity_         (iConfig.getUntrackedParameter<unsigned int>("verbosity", 0)),
  particleId_        (iConfig.getParameter<unsigned int>("particleId")),
  energy_            (iConfig.getParameter<double>("energy")),
  xi_min_            (iConfig.getParameter<double>("xi_min")),
  xi_max_            (iConfig.getParameter<double>("xi_max")),
  theta_x_mean_      (iConfig.getParameter<double>("theta_x_mean")),
  theta_x_sigma_     (iConfig.getParameter<double>("theta_x_sigma")),
  theta_y_mean_      (iConfig.getParameter<double>("theta_y_mean")),
  theta_y_sigma_     (iConfig.getParameter<double>("theta_y_sigma")),
  nParticlesSector45_(iConfig.getParameter<unsigned int>("nParticlesSector45")),
  nParticlesSector56_(iConfig.getParameter<unsigned int>("nParticlesSector56")),
  engine_(nullptr)
{
  produces<HepMCProduct>("unsmeared");
}

//----------------------------------------------------------------------------------------------------

void RandomXiThetaGunProducer::produce(edm::Event &e, const edm::EventSetup& es)
{
  // get conditions
  edm::Service<edm::RandomNumberGenerator> rng;
  engine_ = &rng->getEngine(e.streamID());

  ESHandle<HepPDT::ParticleDataTable> pdgTable;
  es.getData(pdgTable);

  // prepare HepMC event
  HepMC::GenEvent *fEvt = new HepMC::GenEvent();
  fEvt->set_event_number(e.id().event());

  HepMC::GenVertex *vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0., 0.));
  fEvt->add_vertex(vtx);

  const HepPDT::ParticleData *pData = pdgTable->particle(HepPDT::ParticleID(particleId_));
  double mass = pData->mass().value();

  // generate particles
  unsigned int barcode = 0;

  for (unsigned int i = 0; i < nParticlesSector45_; ++i)
    generateParticle(+1., mass, ++barcode, vtx);

  for (unsigned int i = 0; i < nParticlesSector56_; ++i)
    generateParticle(-1., mass, ++barcode, vtx);

  // save output
  std::unique_ptr<HepMCProduct> output(new HepMCProduct()) ;
  output->addHepMCData(fEvt);
  e.put(std::move(output), "unsmeared");
}

//----------------------------------------------------------------------------------------------------

void RandomXiThetaGunProducer::generateParticle(double z_sign, double mass, unsigned int barcode,
  HepMC::GenVertex *vtx) const
{
  const double xi = CLHEP::RandFlat::shoot(engine_, xi_min_, xi_max_);
  const double theta_x = CLHEP::RandGauss::shoot(engine_, theta_x_mean_, theta_x_sigma_);
  const double theta_y = CLHEP::RandGauss::shoot(engine_, theta_y_mean_, theta_y_sigma_);

  if (verbosity_)
    LogInfo("RandomXiThetaGunProducer") << "xi = " << xi << ", theta_x = " << theta_x
      << ", theta_y" << theta_y << ", z_sign = " << z_sign;

  const double cos_theta = sqrt(1. - theta_x*theta_x - theta_y*theta_y);

  const double p_nom = sqrt(energy_*energy_ - mass*mass);
  const double p = p_nom * (1. - xi);
  const double e = sqrt(p*p + mass*mass);

  HepMC::FourVector momentum(
    p * theta_x,
    p * theta_y,
    z_sign * p * cos_theta,
    e
  );

  HepMC::GenParticle* particle = new HepMC::GenParticle(momentum, particleId_, 1);
  particle->suggest_barcode(barcode);
  vtx->add_particle_out(particle);
}
