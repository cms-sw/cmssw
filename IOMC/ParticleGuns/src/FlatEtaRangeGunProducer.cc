/** \class FlatEtaRangeGunProducer
 *
 * Particle gun that shoots particles flat in eta, phi, and energy, with
 * symmetric +/- eta coverage and optional minimum dR separation.
 *
 * Derived from FlatRandomEGunProducer.
 */

#include <memory>
#include <ostream>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

#include "HepPDT/ParticleDataTable.hh"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "FlatRandomEGunProducer.h"

namespace edm {

  class FlatEtaRangeGunProducer : public FlatRandomEGunProducer {
  public:
    FlatEtaRangeGunProducer(const ParameterSet&);
    ~FlatEtaRangeGunProducer() override = default;

    // added by Claude: cms_pepr migration from pepr_15_1_0
    // Required in pre1: framework validates params against fillDescriptions.
    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    // end added by Claude

  private:
    void produce(Event&, const EventSetup&) override;

  protected:
    // the number of particles to shoot
    int nParticles_;

    // flag that denotes that exactly the particles defined by fPartIDs should be shot,
    // with that order and quantity
    bool exactShoot_;

    // flag that denotes whether a random number of particles in the range [1, nParticles_] is shot
    bool randomShoot_;

    double minDr_;

    // debug flag
    bool debug_;
  };

  FlatEtaRangeGunProducer::FlatEtaRangeGunProducer(const ParameterSet& params)
      : FlatRandomEGunProducer(params),
        nParticles_(params.getParameter<ParameterSet>("PGunParameters").getParameter<int>("nParticles")),
        exactShoot_(params.getParameter<ParameterSet>("PGunParameters").getParameter<bool>("exactShoot")),
        randomShoot_(params.getParameter<ParameterSet>("PGunParameters").getParameter<bool>("randomShoot")),
        minDr_(params.getParameter<ParameterSet>("PGunParameters").getUntrackedParameter<double>("minDr", -1.)),
        debug_(params.getUntrackedParameter<bool>("debug")) {}

  // added by Claude: cms_pepr migration from pepr_15_1_0
  // Extends FlatRandomEGunProducer's descriptions with the pepr-added params
  // (nParticles, exactShoot, randomShoot, minDr, debug). Without this, pre1
  // rejects those params as "illegal" during EventProcessor construction.
  void FlatEtaRangeGunProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    ParameterSetDescription pgunParams;
    pgunParams.add<double>("MinE");
    pgunParams.add<double>("MaxE");
    pgunParams.add<int>("nParticles");
    pgunParams.add<bool>("exactShoot");
    pgunParams.add<bool>("randomShoot");
    pgunParams.addUntracked<double>("minDr", -1.);
    BaseFlatGunProducer::fillDescription(desc, pgunParams);
    desc.addUntracked<bool>("debug", false);
    descriptions.addDefault(desc);
  }
  // end added by Claude

  void FlatEtaRangeGunProducer::produce(Event& event, const EventSetup& setup) {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &(rng->getEngine(event.streamID()));

    if (debug_) {
      LogDebug("FlatEtaRangeGunProducer") << " : Begin New Event Generation" << std::endl;
    }

    // create a new event to fill
    auto* genEvent = new HepMC::GenEvent();

    // determine the number of particles to shoot
    int n = 0;
    if (exactShoot_) {
      n = (int)fPartIDs.size();
    } else if (randomShoot_) {
      n = CLHEP::RandFlat::shoot(engine, 1, nParticles_ + 1);
    } else {
      n = nParticles_;
    }

    std::vector<math::XYZVector> previousp4;
    int particle_counter = 0;
    // shoot particles: n for positive and n for negative eta
    for (int i = 0; i < 2 * n; i++) {
      // obtain kinematics
      int id = fPartIDs[exactShoot_ ? particle_counter : CLHEP::RandFlat::shoot(engine, 0, fPartIDs.size())];
      particle_counter++;
      if (particle_counter >= n)
        particle_counter = 0;

      const HepPDT::ParticleData* pData = fPDGTable->particle(HepPDT::ParticleID(abs(id)));
      double eta = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta);
      if (i < n)
        eta *= -1;
      double phi = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);

      double e = CLHEP::RandFlat::shoot(engine, fMinE, fMaxE);
      double m = pData->mass().value();
      double p = sqrt(e * e - m * m);
      math::XYZVector pVec = p * math::XYZVector(cos(phi), sin(phi), sinh(eta)).unit();

      // enforce optional minimum dR separation
      if (minDr_ > 0) {
        bool isgood = true;
        for (const auto& ppvec : previousp4) {
          double drsq = reco::deltaR2(pVec, ppvec);
          if (drsq < minDr_) {
            isgood = false;
            break;
          }
        }
        if (!isgood)
          continue;
      }

      previousp4.push_back(pVec);

      HepMC::GenVertex* vtx = new HepMC::GenVertex(HepMC::FourVector(0, 0, 0, 0));

      // create the GenParticle
      HepMC::FourVector fVec(pVec.x(), pVec.y(), pVec.z(), e);
      HepMC::GenParticle* particle = new HepMC::GenParticle(fVec, id, 1);
      particle->suggest_barcode(i + 1);

      // add the particle to the vertex and the vertex to the event
      vtx->add_particle_out(particle);
      genEvent->add_vertex(vtx);

      if (debug_) {
        vtx->print();
        particle->print();
      }
    }

    // fill event attributes
    genEvent->set_event_number(event.id().event());
    genEvent->set_signal_process_id(20);

    if (debug_) {
      genEvent->print();
    }

    // store outputs
    std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
    BProduct->addHepMCData(genEvent);
    event.put(std::move(BProduct), "unsmeared");
    auto genEventInfo = std::make_unique<GenEventInfoProduct>(genEvent);
    event.put(std::move(genEventInfo));

    if (debug_) {
      LogDebug("FlatEtaRangeGunProducer") << " : Event Generation Done " << std::endl;
    }
  }

}  // namespace edm

DEFINE_FWK_MODULE(edm::FlatEtaRangeGunProducer);
