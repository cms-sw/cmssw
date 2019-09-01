#include <ostream>

#include "IOMC/ParticleGuns/interface/GaussRandomPThetaGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

//#define DebugLog

namespace CLHEP {
  class HepRandomEngine;
}

using namespace edm;

GaussRandomPThetaGunProducer::GaussRandomPThetaGunProducer(const edm::ParameterSet& pset)
    : FlatBaseThetaGunProducer(pset) {
  edm::ParameterSet defpset;
  edm::ParameterSet pgun_params = pset.getParameter<edm::ParameterSet>("PGunParameters");

  // doesn't seem necessary to check if pset is empty - if this
  // is the case, default values will be taken for params
  fMeanP = pgun_params.getParameter<double>("MeanP");
  fSigmaP = pgun_params.getParameter<double>("SigmaP");
  fMeanTheta = 0.5 * (fMinTheta + fMaxTheta);
  fSigmaTheta = 0.5 * (fMaxTheta - fMinTheta);

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

GaussRandomPThetaGunProducer::~GaussRandomPThetaGunProducer() {}

void GaussRandomPThetaGunProducer::produce(edm::Event& e, const edm::EventSetup& es) {
#ifdef DebugLog
  if (fVerbosity >= 0)
    std::cout << "GaussRandomPThetaGunProducer : Begin New Event Generation\n";
#endif
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  // event loop (well, another step in it...)

  // no need to clean up GenEvent memory - done in HepMCProduct

  // here re-create fEvt (memory)
  //
  fEvt = new HepMC::GenEvent();

  // now actualy, cook up the event from PDGTable and gun parameters
  //

  // 1st, primary vertex
  //
  HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0.));

  // loop over particles
  //
  int barcode = 1;
  for (unsigned int ip = 0; ip < fPartIDs.size(); ip++) {
    double mom = CLHEP::RandGaussQ::shoot(engine, fMeanP, fSigmaP);
    double theta = CLHEP::RandGaussQ::shoot(engine, fMeanTheta, fSigmaTheta);
    double phi = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
    int PartID = fPartIDs[ip];
    const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
    double mass = PData->mass().value();
    double energy = sqrt(mom * mom + mass * mass);
    double px = mom * sin(theta) * cos(phi);
    double py = mom * sin(theta) * sin(phi);
    double pz = mom * cos(theta);

    HepMC::FourVector p(px, py, pz, energy);
    HepMC::GenParticle* Part = new HepMC::GenParticle(p, PartID, 1);
    Part->suggest_barcode(barcode);
    barcode++;
    Vtx->add_particle_out(Part);

    if (fAddAntiParticle) {
      HepMC::FourVector ap(-px, -py, -pz, energy);
      int APartID = -PartID;
      if (PartID == 22 || PartID == 23) {
        APartID = PartID;
      }
      HepMC::GenParticle* APart = new HepMC::GenParticle(ap, APartID, 1);
      APart->suggest_barcode(barcode);
      barcode++;
      Vtx->add_particle_out(APart);
    }
  }
  fEvt->add_vertex(Vtx);
  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(20);

#ifdef DebugLog
  if (fVerbosity >= 0)
    fEvt->print();
#endif

  std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
  BProduct->addHepMCData(fEvt);
  e.put(std::move(BProduct), "unsmeared");

  std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(std::move(genEventInfo));

#ifdef DebugLog
  if (fVerbosity >= 0)
    std::cout << "GaussRandomPThetaGunProducer : Event Generation Done\n";
#endif
}
