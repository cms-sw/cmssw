/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "IOMC/EventVertexGenerators/interface/BeamDivergenceVtxGenerator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include <CLHEP/Random/RandGauss.h>

//----------------------------------------------------------------------------------------------------

BeamDivergenceVtxGenerator::BeamDivergenceVtxGenerator(const edm::ParameterSet& iConfig) :
  sourceToken_(consumes<edm::HepMCProduct>( iConfig.getParameter<edm::InputTag>("src"))),
  simulateVertex_        (iConfig.getParameter<bool>("simulateVertex")),
  simulateBeamDivergence_(iConfig.getParameter<bool>("simulateBeamDivergence"))
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable())
    throw cms::Exception("Configuration")
      << "The BeamDivergenceVtxGenerator requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file. \n"
         "You must add the service\n"
         "in the configuration file or remove the modules that require it.";

  produces<edm::HepMCProduct>();
}

//----------------------------------------------------------------------------------------------------

void
BeamDivergenceVtxGenerator::produce(edm::Event& iEvent, const edm::EventSetup &iSetup)
{
  // get random engine
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* rnd = &(rng->getEngine(iEvent.streamID()));

  // get conditions
  edm::ESHandle<CTPPSBeamParameters> hBeamParameters;
  iSetup.get<CTPPSBeamParametersRcd>().get(hBeamParameters);

  // get input
  edm::Handle<edm::HepMCProduct> hepUnsmearedMCEvt;
  iEvent.getByToken(sourceToken_, hepUnsmearedMCEvt);

  // prepare output
  HepMC::GenEvent* genevt = new HepMC::GenEvent(*hepUnsmearedMCEvt->GetEvent());
  std::unique_ptr<edm::HepMCProduct> pEvent(new edm::HepMCProduct(genevt));

  // apply vertex smearing
  if (simulateVertex_) {
    // NB: the separtion between effective offsets in LHC sectors 45 and 56 cannot be applied, thus the values for 45 are used
    const double vtx_x = hBeamParameters->getVtxOffsetX45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevX();
    const double vtx_y = hBeamParameters->getVtxOffsetY45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevY();
    const double vtx_z = hBeamParameters->getVtxOffsetZ45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevZ();

    HepMC::FourVector shift(vtx_x*1E1, vtx_y*1E1, vtx_z*1E1, 0.);  // conversions: cm to mm
    pEvent->applyVtxGen(&shift);
  }

  // apply beam divergence
  if (simulateBeamDivergence_) {
    const double bd_x_45 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceX45();
    const double bd_x_56 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceX56();

    const double bd_y_45 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceY45();
    const double bd_y_56 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceY56();

    for (HepMC::GenEvent::particle_iterator part = genevt->particles_begin(); part != genevt->particles_end(); ++part) {
      const HepMC::FourVector mom = (*part)->momentum();

      // TODO: this is an oversimplified implemetation
      // the TOTEM smearing module should be taken as reference

      double th_x = mom.x() / mom.z();
      double th_y = mom.y() / mom.z();

      if (mom.z() > 0.)
      {
        th_x += bd_x_45;
        th_y += bd_y_45;
      } else {
        th_x += bd_x_56;
        th_y += bd_y_56;
      }

      // calculate consistent p_z component
      const double sign = (mom.z() > 0.) ? 1. : -1.;
      const double p_z = sign * mom.rho() / sqrt(1. + th_x*th_x + th_y*th_y);

      // set smeared momentum
      (*part)->set_momentum(HepMC::FourVector(p_z * th_x, p_z * th_y, p_z, mom.e()));
    }
  }

  // save output
  iEvent.put(std::move(pEvent));
}

void
BeamDivergenceVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("generator", "unsmeared"))
    ->setComment("input collection where to retrieve outgoing particles kinematics to be smeared");
  desc.add<bool>("simulateBeamDivergence", true)->setComment("account for the beam angular divergence?");
  desc.add<bool>("simulateVertex", true)->setComment("account for the vertex transverse smearing?");

  descriptions.add("beamDivergenceVtxGenerator", desc);
}

