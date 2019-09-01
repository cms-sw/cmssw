#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomPtAndDxyGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

FlatRandomPtAndDxyGunProducer::FlatRandomPtAndDxyGunProducer(const ParameterSet& pset) : BaseFlatGunProducer(pset) {
  ParameterSet defpset;
  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");

  fMinPt = pgun_params.getParameter<double>("MinPt");
  fMaxPt = pgun_params.getParameter<double>("MaxPt");
  dxyMin_ = pgun_params.getParameter<double>("dxyMin");
  dxyMax_ = pgun_params.getParameter<double>("dxyMax");
  lxyMax_ = pgun_params.getParameter<double>("LxyMax");
  lzMax_ = pgun_params.getParameter<double>("LzMax");
  ConeRadius_ = pgun_params.getParameter<double>("ConeRadius");
  ConeH_ = pgun_params.getParameter<double>("ConeH");
  DistanceToAPEX_ = pgun_params.getParameter<double>("DistanceToAPEX");

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

FlatRandomPtAndDxyGunProducer::~FlatRandomPtAndDxyGunProducer() {
  // no need to cleanup GenEvent memory - done in HepMCProduct
}

void FlatRandomPtAndDxyGunProducer::produce(Event& e, const EventSetup& es) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  if (fVerbosity > 0) {
    cout << " FlatRandomPtAndDxyGunProducer : Begin New Event Generation" << endl;
  }
  // event loop (well, another step in it...)

  // no need to clean up GenEvent memory - done in HepMCProduct
  //

  // here re-create fEvt (memory)
  //
  fEvt = new HepMC::GenEvent();

  // now actualy, cook up the event from PDGTable and gun parameters
  int barcode = 1;
  for (unsigned int ip = 0; ip < fPartIDs.size(); ++ip) {
    double phi_vtx = 0;
    double dxy = 0;
    double pt = 0;
    double eta = 0;
    double px = 0;
    double py = 0;
    double pz = 0;
    double vx = 0;
    double vy = 0;
    double vz = 0;
    double lxy = 0;

    bool passLoop = false;
    while (not passLoop) {
      bool passLxy = false;
      bool passLz = false;
      phi_vtx = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
      dxy = CLHEP::RandFlat::shoot(engine, dxyMin_, dxyMax_);
      float dxysign = CLHEP::RandFlat::shoot(engine, -1, 1);
      if (dxysign < 0)
        dxy = -dxy;
      pt = CLHEP::RandFlat::shoot(engine, fMinPt, fMaxPt);
      px = pt * cos(phi_vtx);
      py = pt * sin(phi_vtx);
      for (int i = 0; i < 10000; i++) {
        vx = CLHEP::RandFlat::shoot(engine, -lxyMax_, lxyMax_);
        vy = (pt * dxy + vx * py) / px;
        lxy = sqrt(vx * vx + vy * vy);
        if (lxy < abs(lxyMax_) and (vx * px + vy * py) > 0) {
          passLxy = true;
          break;
        }
      }
      eta = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta);
      pz = pt * sinh(eta);
      //vz = fabs(fRandomGaussGenerator->fire(0.0, LzWidth_/2.0));
      float ConeTheta = ConeRadius_ / ConeH_;
      for (int j = 0; j < 100; j++) {
        vz = CLHEP::RandFlat::shoot(engine, 0.0, lzMax_);  // this is abs(vz)
        float v0 = vz - DistanceToAPEX_;
        if (v0 <= 0 or lxy * lxy / (ConeTheta * ConeTheta) > v0 * v0) {
          passLz = true;
          break;
        }
      }
      if (pz < 0)
        vz = -vz;
      passLoop = (passLxy and passLz);

      if (passLoop)
        break;
    }

    HepMC::GenVertex* Vtx1 = new HepMC::GenVertex(HepMC::FourVector(vx, vy, vz));

    int PartID = fPartIDs[ip];
    const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
    double mass = PData->mass().value();
    double energy2 = px * px + py * py + pz * pz + mass * mass;
    double energy = sqrt(energy2);
    HepMC::FourVector p(px, py, pz, energy);
    HepMC::GenParticle* Part = new HepMC::GenParticle(p, PartID, 1);
    Part->suggest_barcode(barcode);
    barcode++;
    Vtx1->add_particle_out(Part);
    fEvt->add_vertex(Vtx1);

    if (fAddAntiParticle) {
      HepMC::GenVertex* Vtx2 = new HepMC::GenVertex(HepMC::FourVector(-vx, -vy, -vz));
      HepMC::FourVector ap(-px, -py, -pz, energy);
      int APartID = -PartID;
      if (PartID == 22 || PartID == 23) {
        APartID = PartID;
      }
      HepMC::GenParticle* APart = new HepMC::GenParticle(ap, APartID, 1);
      APart->suggest_barcode(barcode);
      barcode++;
      Vtx2->add_particle_out(APart);
      fEvt->add_vertex(Vtx2);
    }
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
    cout << " FlatRandomPtAndDxyGunProducer : End New Event Generation" << endl;
    fEvt->print();
  }
}
