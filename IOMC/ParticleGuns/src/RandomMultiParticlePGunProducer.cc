#include <ostream>

#include "IOMC/ParticleGuns/interface/RandomMultiParticlePGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CLHEP/Random/RandFlat.h"

//#define DebugLog
using namespace edm;

RandomMultiParticlePGunProducer::RandomMultiParticlePGunProducer(const ParameterSet& pset) : BaseFlatGunProducer(pset) {
  ParameterSet pgunParams = pset.getParameter<ParameterSet>("PGunParameters");
  fProbParticle_ = pgunParams.getParameter<std::vector<double> >("ProbParts");
  fProbP_ = pgunParams.getParameter<std::vector<double> >("ProbP");
  fMinP_ = pgunParams.getParameter<double>("MinP");
  fMaxP_ = pgunParams.getParameter<double>("MaxP");
  if (fProbParticle_.size() != fPartIDs.size())
    throw cms::Exception("Configuration") << "Not all probabilities given for all particle types "
                                          << fProbParticle_.size() << ":" << fPartIDs.size() << " need them to match\n";

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
  fBinsP_ = (int)(fProbP_.size());
  fDeltaP_ = (fMaxP_ - fMinP_) / fBinsP_;
#ifdef DebugLog
  edm::LogVerbatim("IOMC") << "Internal FlatRandomPGun is initialzed for " << fPartIDs.size() << " particles in "
                           << fBinsP_ << " bins within momentum range " << fMinP_ << ":" << fMaxP_;
  for (unsigned int k = 0; k < fPartIDs.size(); ++k)
    edm::LogVerbatim("IOMC") << " [" << k << "] " << fPartIDs[k] << ":" << fProbParticle_[k];
  edm::LogVerbatim("IOMC") << "Momentum distribution is given by";
  for (int k = 0; k < fBinsP_; ++k) {
    double p = fMinP_ + k * fDeltaP_;
    edm::LogVerbatim("IOMC") << " Bin[" << k << "] " << p << ":" << p + fDeltaP_ << " --> " << fProbP_[k];
  }
#endif
  for (unsigned int k = 1; k < fProbParticle_.size(); ++k)
    fProbParticle_[k] += fProbParticle_[k - 1];
  for (unsigned int k = 0; k < fProbParticle_.size(); ++k)
    fProbParticle_[k] /= fProbParticle_[fProbParticle_.size() - 1];
#ifdef DebugLog
  edm::LogVerbatim("IOMC") << "Corrected probabilities for particle type:";
  for (unsigned int k = 0; k < fProbParticle_.size(); ++k)
    edm::LogVerbatim("IOMC") << "  [" << k << "]: " << fProbParticle_[k];
#endif
  for (int k = 1; k < fBinsP_; ++k)
    fProbP_[k] += fProbP_[k - 1];
  for (int k = 0; k < fBinsP_; ++k)
    fProbP_[k] /= fProbP_[fBinsP_ - 1];
#ifdef DebugLog
  edm::LogVerbatim("IOMC") << "Corrected probabilities for momentum:";
  for (int k = 0; k < fBinsP_; ++k) {
    double p = fMinP_ + k * fDeltaP_;
    edm::LogVerbatim("IOMC") << " Bin[" << k << "] " << p << ":" << p + fDeltaP_ << " --> " << fProbP_[k];
  }
#endif
}

void RandomMultiParticlePGunProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

#ifdef DebugLog
  if (fVerbosity > 0)
    edm::LogVerbatim("IOMC") << "RandomMultiParticlePGunProducer: "
                             << "Begin New Event Generation";
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
  double minP(fMinP_);
  double r2 = CLHEP::RandFlat::shoot(engine, 0., 1.);
  for (int ip = 0; ip < fBinsP_; ip++) {
    if (r2 <= fProbP_[ip]) {
      minP = fMinP_ + ip * fDeltaP_;
      break;
    }
  }
  double maxP = minP + fDeltaP_;
#ifdef DebugLog
  if (fVerbosity > 0)
    edm::LogVerbatim("IOMC") << "Random " << r1 << " PartID " << PartID << " and for p " << r2 << " range " << minP
                             << ":" << maxP;
#endif
  double mom = CLHEP::RandFlat::shoot(engine, minP, maxP);
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
    edm::LogVerbatim("IOMC") << " RandomMultiParticlePGunProducer : Event Generation Done ";
#endif
}
