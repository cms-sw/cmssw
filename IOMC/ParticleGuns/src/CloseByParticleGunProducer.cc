#include <ostream>

#include "IOMC/ParticleGuns/interface/CloseByParticleGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

CloseByParticleGunProducer::CloseByParticleGunProducer(const ParameterSet& pset) : BaseFlatGunProducer(pset) {
  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");
  fControlledByEta = pgun_params.getParameter<bool>("ControlledByEta");
  fEnMax = pgun_params.getParameter<double>("EnMax");
  fEnMin = pgun_params.getParameter<double>("EnMin");
  fMaxEnSpread = pgun_params.getParameter<bool>("MaxEnSpread");
  if (fControlledByEta) {
    fEtaMax = pgun_params.getParameter<double>("MaxEta");
    fEtaMin = pgun_params.getParameter<double>("MinEta");
    if (fEtaMax <= fEtaMin)
      LogError("CloseByParticleGunProducer") << " Please fix MinEta and MaxEta values in the configuration";
  } else {
    fRMax = pgun_params.getParameter<double>("RMax");
    fRMin = pgun_params.getParameter<double>("RMin");
    if (fRMax <= fRMin)
      LogError("CloseByParticleGunProducer") << " Please fix RMin and RMax values in the configuration";
  }
  fZMax = pgun_params.getParameter<double>("ZMax");
  fZMin = pgun_params.getParameter<double>("ZMin");
  fDelta = pgun_params.getParameter<double>("Delta");
  fPhiMin = pgun_params.getParameter<double>("MinPhi");
  fPhiMax = pgun_params.getParameter<double>("MaxPhi");
  fPointing = pgun_params.getParameter<bool>("Pointing");
  fOverlapping = pgun_params.getParameter<bool>("Overlapping");
  fRandomShoot = pgun_params.getParameter<bool>("RandomShoot");
  fNParticles = pgun_params.getParameter<int>("NParticles");
  fPartIDs = pgun_params.getParameter<vector<int> >("PartID");

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

CloseByParticleGunProducer::~CloseByParticleGunProducer() {
  // no need to cleanup GenEvent memory - done in HepMCProduct
}

void CloseByParticleGunProducer::produce(Event& e, const EventSetup& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  if (fVerbosity > 0) {
    LogDebug("CloseByParticleGunProducer") << " CloseByParticleGunProducer : Begin New Event Generation" << endl;
  }
  fEvt = new HepMC::GenEvent();

  int barcode = 1;
  unsigned int numParticles = fRandomShoot ? CLHEP::RandFlat::shoot(engine, 1, fNParticles) : fNParticles;

  double phi = CLHEP::RandFlat::shoot(engine, fPhiMin, fPhiMax);
  double fZ = CLHEP::RandFlat::shoot(engine, fZMin, fZMax);
  double fR;

  if (!fControlledByEta) {
    fR = CLHEP::RandFlat::shoot(engine, fRMin, fRMax);
  } else {
    double fEta = CLHEP::RandFlat::shoot(engine, fEtaMin, fEtaMax);
    fR = (fZ / sinh(fEta));
  }

  double tmpPhi = phi;
  double tmpR = fR;

  // Loop over particles
  for (unsigned int ip = 0; ip < numParticles; ++ip) {
    if (fOverlapping) {
      fR = CLHEP::RandFlat::shoot(engine, tmpR - fDelta, tmpR + fDelta);
      phi = CLHEP::RandFlat::shoot(engine, tmpPhi - fDelta / fR, tmpPhi + fDelta / fR);
    } else
      phi += fDelta / fR;

    double fEn;
    if (numParticles > 1 && fMaxEnSpread)
      fEn = fEnMin + ip * (fEnMax - fEnMin) / (numParticles - 1);
    else
      fEn = CLHEP::RandFlat::shoot(engine, fEnMin, fEnMax);

    int partIdx = CLHEP::RandFlat::shoot(engine, 0, fPartIDs.size());
    int PartID = fPartIDs[partIdx];
    const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
    double mass = PData->mass().value();
    double mom2 = fEn * fEn - mass * mass;
    double mom = 0.;
    if (mom2 > 0.) {
      mom = sqrt(mom2);
    }
    double px = 0.;
    double py = 0.;
    double pz = mom;
    double energy = fEn;

    // Compute Vertex Position
    double x = fR * cos(phi);
    double y = fR * sin(phi);
    constexpr double c = 2.99792458e+1;  // cm/ns
    double timeOffset = sqrt(x * x + y * y + fZ * fZ) / c * ns * c_light;
    HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(x * cm, y * cm, fZ * cm, timeOffset));

    HepMC::FourVector p(px, py, pz, energy);
    // If we are requested to be pointing to (0,0,0), correct the momentum direction
    if (fPointing) {
      math::XYZVector direction(x, y, fZ);
      math::XYZVector momentum = direction.unit() * mom;
      p.setX(momentum.x());
      p.setY(momentum.y());
      p.setZ(momentum.z());
    }
    HepMC::GenParticle* Part = new HepMC::GenParticle(p, PartID, 1);
    Part->suggest_barcode(barcode);
    barcode++;

    Vtx->add_particle_out(Part);

    if (fVerbosity > 0) {
      Vtx->print();
      Part->print();
    }
    fEvt->add_vertex(Vtx);
  }

  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(20);

  if (fVerbosity > 0) {
    fEvt->print();
  }

  unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
  BProduct->addHepMCData(fEvt);
  e.put(std::move(BProduct), "unsmeared");

  unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(std::move(genEventInfo));

  if (fVerbosity > 0) {
    LogDebug("CloseByParticleGunProducer") << " CloseByParticleGunProducer : Event Generation Done " << endl;
  }
}
