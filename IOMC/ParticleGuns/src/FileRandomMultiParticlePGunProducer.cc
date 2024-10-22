#include <ostream>

#include "IOMC/ParticleGuns/interface/FileRandomMultiParticlePGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CLHEP/Random/RandFlat.h"

using namespace edm;

const unsigned int np = 6;
const unsigned int kfactor = 100;

FileRandomMultiParticlePGunProducer::FileRandomMultiParticlePGunProducer(const ParameterSet& pset)
    : BaseFlatGunProducer(pset) {
  ParameterSet pgunParams = pset.getParameter<ParameterSet>("PGunParameters");
  fMinP_ = pgunParams.getParameter<double>("MinP");
  fMaxP_ = pgunParams.getParameter<double>("MaxP");
  edm::FileInPath fp = pgunParams.getParameter<edm::FileInPath>("FileName");
  std::string file = fp.fullPath();

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
  edm::LogVerbatim("ParticleGun") << "FileRandomMultiParticlePGun is initialzed with i/p file " << file
                                  << " and use momentum range " << fMinP_ << ":" << fMaxP_;

  if (fPartIDs.size() != np)
    throw cms::Exception("ParticleGun") << "Invalid list of partices: " << fPartIDs.size() << " should be " << np
                                        << "\n";

  std::ifstream is(file.c_str(), std::ios::in);
  if (!is) {
    throw cms::Exception("Configuration") << "Cannot find the file " << file << "\n";
  } else {
    double xl, xh;
    is >> fPBin_ >> xl >> xh >> fEtaBin_ >> fEtaMin_ >> fEtaBinWidth_;
    fP_.emplace_back(xl);
    edm::LogVerbatim("ParticleGun") << "FileRandomMultiParticlePGun: p: " << fPBin_ << ":" << xl << ":" << xh
                                    << " Eta: " << fEtaBin_ << ":" << fEtaMin_ << ":" << fEtaBinWidth_;
    for (int ip = 0; ip < fPBin_; ++ip) {
      for (int ie = 0; ie < fEtaBin_; ++ie) {
        double totprob(0);
        std::vector<double> prob(np, 0);
        int je;
        is >> xl >> xh >> je >> prob[0] >> prob[1] >> prob[2] >> prob[3] >> prob[4] >> prob[5];
        if (ie == 0)
          fP_.emplace_back(xh);
        for (unsigned int k = 0; k < np; ++k) {
          totprob += prob[k];
          if (k > 0)
            prob[k] += prob[k - 1];
        }
        for (unsigned int k = 0; k < np; ++k)
          prob[k] /= totprob;
        int indx = (ip + 1) * kfactor + ie;
        fProbParticle_[indx] = prob;
        if (fVerbosity > 0)
          edm::LogVerbatim("ParticleGun")
              << "FileRandomMultiParticlePGun [" << ip << "," << ie << ", " << indx << "] Probability " << prob[0]
              << ", " << prob[1] << ", " << prob[2] << ", " << prob[3] << ", " << prob[4] << ", " << prob[5];
      }
    }
    is.close();
  }
}

FileRandomMultiParticlePGunProducer::~FileRandomMultiParticlePGunProducer() {}

void FileRandomMultiParticlePGunProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  if (fVerbosity > 0)
    edm::LogVerbatim("ParticleGun") << "FileRandomMultiParticlePGunProducer: Begin New Event Generation";

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

  // Now p, eta, phi
  double mom = CLHEP::RandFlat::shoot(engine, fMinP_, fMaxP_);
  double eta = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta);
  double phi = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
  int ieta = static_cast<int>((eta - fEtaMin_) / fEtaBinWidth_);
  auto ipp = std::lower_bound(fP_.begin(), fP_.end(), mom);
  if (ipp == fP_.end())
    --ipp;
  int ip = static_cast<int>(ipp - fP_.begin());
  int indx = ip * kfactor + ieta;
  if (fVerbosity > 0)
    edm::LogVerbatim("ParticleGun") << "FileRandomMultiParticlePGunProducer: p " << mom << " Eta " << eta << " Phi "
                                    << phi << " Index " << indx;

  // Now particle id
  //
  int barcode(0), partID(fPartIDs[0]);
  double r1 = CLHEP::RandFlat::shoot(engine, 0., 1.);
  for (unsigned int ip = 0; ip < fPartIDs.size(); ip++) {
    if (r1 <= fProbParticle_[indx][ip])
      break;
    partID = fPartIDs[ip];
  }
  if (fVerbosity > 0)
    edm::LogVerbatim("ParticleGun") << "Random " << r1 << " PartID " << partID;
  const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(partID));
  double mass = PData->mass().value();
  double energy = sqrt(mom * mom + mass * mass);
  double theta = 2. * atan(exp(-eta));
  double px = mom * sin(theta) * cos(phi);
  double py = mom * sin(theta) * sin(phi);
  double pz = mom * cos(theta);

  HepMC::FourVector p(px, py, pz, energy);
  HepMC::GenParticle* Part = new HepMC::GenParticle(p, partID, 1);
  barcode++;
  Part->suggest_barcode(barcode);
  Vtx->add_particle_out(Part);

  fEvt->add_vertex(Vtx);
  fEvt->set_event_number(e.id().event());
  fEvt->set_signal_process_id(20);

  if (fVerbosity > 1)
    fEvt->print();

  std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
  BProduct->addHepMCData(fEvt);
  e.put(std::move(BProduct), "unsmeared");

  std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(std::move(genEventInfo));
  if (fVerbosity > 0)
    edm::LogVerbatim("ParticleGun") << "FileRandomMultiParticlePGunProducer : Event Generation Done";
}
