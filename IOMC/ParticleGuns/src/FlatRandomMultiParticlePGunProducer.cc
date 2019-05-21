#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomMultiParticlePGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CLHEP/Random/RandFlat.h"

//#define DebugLog
using namespace edm;

FlatRandomMultiParticlePGunProducer::FlatRandomMultiParticlePGunProducer(const ParameterSet& pset)
    : BaseFlatGunProducer(pset) {
  ParameterSet pgunParams = pset.getParameter<ParameterSet>("PGunParameters");
  fProbParticle_ = pgunParams.getParameter<std::vector<double> >("ProbParts");
  fMinP_ = pgunParams.getParameter<double>("MinP");
  fMaxP_ = pgunParams.getParameter<double>("MaxP");
  if (fProbParticle_.size() != fPartIDs.size())
    throw cms::Exception("Configuration") << "Not all probabilities given for all particle types "
                                          << fProbParticle_.size() << ":" << fPartIDs.size() << " need them to match\n";

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
#ifdef DebugLog
  std::cout << "Internal FlatRandomPGun is initialzed for " << fPartIDs.size() << " particles in momentum range "
            << fMinP_ << ":" << fMaxP_ << std::endl;
  for (unsigned int k = 0; k < fPartIDs.size(); ++k)
    std::cout << " [" << k << "] " << fPartIDs[k] << ":" << fProbParticle_[k];
  std::cout << std::endl;
#endif
  for (unsigned int k = 1; k < fProbParticle_.size(); ++k)
    fProbParticle_[k] += fProbParticle_[k - 1];
  for (unsigned int k = 0; k < fProbParticle_.size(); ++k)
    fProbParticle_[k] /= fProbParticle_[fProbParticle_.size() - 1];
#ifdef DebugLog
  std::cout << "Corrected probabilities:";
  for (unsigned int k = 0; k < fProbParticle_.size(); ++k)
    std::cout << "  " << fProbParticle_[k];
  std::cout << std::endl;
#endif
}

FlatRandomMultiParticlePGunProducer::~FlatRandomMultiParticlePGunProducer() {}

void FlatRandomMultiParticlePGunProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

#ifdef DebugLog
  if (fVerbosity > 0)
    std::cout << "FlatRandomMultiParticlePGunProducer: Begin New Event Generation" << std::endl;
#endif

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
  int barcode(0), PartID(fPartIDs[0]);
  double r1 = CLHEP::RandFlat::shoot(engine, 0., 1.);
  for (unsigned int ip = 0; ip < fPartIDs.size(); ip++) {
    if (r1 <= fProbParticle_[ip]) {
      PartID = fPartIDs[ip];
      break;
    }
  }
#ifdef DebugLog
  if (fVerbosity > 0)
    std::cout << "Random " << r1 << " PartID " << PartID << std::endl;
#endif
  double mom = CLHEP::RandFlat::shoot(engine, fMinP_, fMaxP_);
  double eta = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta);
  double phi = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
  const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
  double mass = PData->mass().value();
  double energy = sqrt(mom * mom + mass * mass);
  double theta = 2. * atan(exp(-eta));
  double px = mom * sin(theta) * cos(phi);
  double py = mom * sin(theta) * sin(phi);
  double pz = mom * cos(theta);

  HepMC::FourVector p(px, py, pz, energy);
  HepMC::GenParticle* Part = new HepMC::GenParticle(p, PartID, 1);
  barcode++;
  Part->suggest_barcode(barcode);
  Vtx->add_particle_out(Part);

  if (fAddAntiParticle) {
    HepMC::FourVector ap(-px, -py, -pz, energy);
    int APartID = (PartID == 22 || PartID == 23) ? PartID : -PartID;
    HepMC::GenParticle* APart = new HepMC::GenParticle(ap, APartID, 1);
    barcode++;
    APart->suggest_barcode(barcode);
    Vtx->add_particle_out(APart);
  }

  fEvt->add_vertex(Vtx);
  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(20);

#ifdef DebugLog
  if (fVerbosity > 0)
    fEvt->print();
#endif

  std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
  BProduct->addHepMCData(fEvt);
  e.put(std::move(BProduct), "unsmeared");

  std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(std::move(genEventInfo));
#ifdef DebugLog
  if (fVerbosity > 0)
    std::cout << " FlatRandomMultiParticlePGunProducer : Event Generation Done " << std::endl;
#endif
}
