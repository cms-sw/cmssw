#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <ostream>
#include <vector>

#include "BaseFlatGunProducer.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <CLHEP/Units/SystemOfUnits.h>

namespace edm {

  class CloseByParticleGunProducer : public BaseFlatGunProducer {
  public:
    explicit CloseByParticleGunProducer(const ParameterSet&);
    ~CloseByParticleGunProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(Event& e, const EventSetup& es) override;

    bool fControlledByEta = false;
    bool fControlledByREta = false;
    double fVarMin = 0.;
    double fVarMax = 0.;
    double fEtaMin = 0.;
    double fEtaMax = 0.;
    double fRMin = 0.;
    double fRMax = 0.;
    double fZMin = 0.;
    double fZMax = 0.;
    double fDelta = 0.;
    double fPhiMin = 0.;
    double fPhiMax = 0.;
    double fTMin = 0.;
    double fTMax = 0.;
    double fOffsetFirst = 0.;
    double log_fVarMin = 0.;
    double log_fVarMax = 0.;
    int fNParticles = 0;
    bool fLogSpacedVar = false;
    bool fMaxVarSpread = false;
    bool fFlatPtGeneration = false;
    bool fPointing = false;
    bool fOverlapping = false;
    bool fRandomShoot = false;
    bool fUseDeltaT = false;
    std::vector<int> fPartIDs;

    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_fieldToken;
  };

  CloseByParticleGunProducer::CloseByParticleGunProducer(const ParameterSet& pset)
      : BaseFlatGunProducer(pset), m_fieldToken(esConsumes()) {
    ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");
    fControlledByEta = pgun_params.getParameter<bool>("ControlledByEta");
    fControlledByREta = pgun_params.getParameter<bool>("ControlledByREta");
    if (fControlledByEta and fControlledByREta) {
      throw cms::Exception("CloseByParticleGunProducer")
          << " Conflicting configuration, cannot have both ControlledByEta and ControlledByREta ";
    }

    fVarMax = pgun_params.getParameter<double>("VarMax");
    fVarMin = pgun_params.getParameter<double>("VarMin");
    fMaxVarSpread = pgun_params.getParameter<bool>("MaxVarSpread");
    fLogSpacedVar = pgun_params.getParameter<bool>("LogSpacedVar");
    fFlatPtGeneration = pgun_params.getParameter<bool>("FlatPtGeneration");

    if (fVarMin < 1. and !fFlatPtGeneration) {
      throw cms::Exception("CloseByParticleGunProducer")
          << " Please choose a minimum energy greater than 1 GeV, otherwise time "
             "information may be invalid or not reliable";
    }

    if (fVarMin <= 0. and fLogSpacedVar) {
      throw cms::Exception("CloseByParticleGunProducer") << " Minimum energy must be greater than zero for log spacing";
    } else {
      log_fVarMin = std::log(fVarMin);
      log_fVarMax = std::log(fVarMax);
    }

    if (fControlledByEta or fControlledByREta) {
      fEtaMax = pgun_params.getParameter<double>("MaxEta");
      fEtaMin = pgun_params.getParameter<double>("MinEta");
      if (fEtaMax <= fEtaMin) {
        throw cms::Exception("CloseByParticleGunProducer")
            << " Please fix MinEta and MaxEta values in the configuration";
      }
    }

    if (!fControlledByEta) {
      fRMax = pgun_params.getParameter<double>("RMax");
      fRMin = pgun_params.getParameter<double>("RMin");
      if (fRMax <= fRMin) {
        throw cms::Exception("CloseByParticleGunProducer") << " Please fix RMin and RMax values in the configuration";
      }
    }

    if (!fControlledByREta) {
      fZMax = pgun_params.getParameter<double>("ZMax");
      fZMin = pgun_params.getParameter<double>("ZMin");
      if (fZMax <= fZMin) {
        throw cms::Exception("CloseByParticleGunProducer") << " Please fix ZMin and ZMax values in the configuration";
      }
    }

    fDelta = pgun_params.getParameter<double>("Delta");
    fPhiMin = pgun_params.getParameter<double>("MinPhi");
    fPhiMax = pgun_params.getParameter<double>("MaxPhi");
    fPointing = pgun_params.getParameter<bool>("Pointing");
    fOverlapping = pgun_params.getParameter<bool>("Overlapping");

    if (fFlatPtGeneration and !fPointing) {
      throw cms::Exception("CloseByParticleGunProducer") << " Can't generate non pointing FlatPt samples; please "
                                                            "disable FlatPt generation or generate pointing sample";
    }

    fRandomShoot = pgun_params.getParameter<bool>("RandomShoot");
    fNParticles = pgun_params.getParameter<int>("NParticles");
    fPartIDs = pgun_params.getParameter<std::vector<int>>("PartID");

    fUseDeltaT = pgun_params.getParameter<bool>("UseDeltaT");
    fTMax = pgun_params.getParameter<double>("TMax");
    fTMin = pgun_params.getParameter<double>("TMin");
    if (fTMax <= fTMin) {
      throw cms::Exception("CloseByParticleGunProducer") << " Please fix TMin and TMax values in the configuration";
    }

    fOffsetFirst = pgun_params.getParameter<double>("OffsetFirst");

    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
  }

  CloseByParticleGunProducer::~CloseByParticleGunProducer() {
    // GenEvent memory is owned by HepMCProduct.
  }

  void CloseByParticleGunProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("AddAntiParticle", false);

    edm::ParameterSetDescription psd0;
    psd0.add<bool>("ControlledByEta", false);
    psd0.add<bool>("ControlledByREta", false);
    psd0.add<double>("Delta", 10.);
    psd0.add<double>("VarMax", 200.);
    psd0.add<double>("VarMin", 25.);
    psd0.add<bool>("MaxVarSpread", false);
    psd0.add<bool>("LogSpacedVar", false);
    psd0.add<bool>("FlatPtGeneration", false);
    psd0.add<double>("MaxEta", 2.7);
    psd0.add<double>("MaxPhi", std::numbers::pi);
    psd0.add<double>("MinEta", 1.7);
    psd0.add<double>("MinPhi", -std::numbers::pi);
    psd0.add<int>("NParticles", 2);
    psd0.add<bool>("Overlapping", false);
    psd0.add<std::vector<int>>("PartID", {22});
    psd0.add<bool>("Pointing", true);
    psd0.add<double>("RMax", 120.);
    psd0.add<double>("RMin", 60.);
    psd0.add<bool>("RandomShoot", false);
    psd0.add<double>("ZMax", 321.);
    psd0.add<double>("ZMin", 320.);
    psd0.add<bool>("UseDeltaT", false);
    psd0.add<double>("TMin", 0.);
    psd0.add<double>("TMax", 0.05);
    psd0.add<double>("OffsetFirst", 0.);

    desc.add<edm::ParameterSetDescription>("PGunParameters", psd0);
    desc.addUntracked<int>("Verbosity", 0);
    desc.addUntracked<unsigned int>("firstRun", 1);
    desc.add<std::string>("psethack", "random particles in phi and r windows");

    descriptions.add("CloseByParticleGunProducer", desc);
  }

  void CloseByParticleGunProducer::produce(Event& e, const EventSetup& es) {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    if (fVerbosity > 0) {
      LogDebug("CloseByParticleGunProducer") << " CloseByParticleGunProducer : Begin New Event Generation" << std::endl;
    }

    fEvt = new HepMC::GenEvent();

    auto const& field = es.getData(m_fieldToken);
    const double bz = field.inTesla({0.f, 0.f, 0.f}).z();

    int barcode = 1;
    const unsigned int numParticles =
        fRandomShoot ? static_cast<unsigned int>(CLHEP::RandFlat::shoot(engine, 1, fNParticles + 1))
                     : static_cast<unsigned int>(fNParticles);

    double phi = CLHEP::RandFlat::shoot(engine, fPhiMin, fPhiMax);
    double fZ = 0.;
    double fR = 0.;
    double fEta = 0.;
    double fT = 0.;

    if (!fControlledByREta) {
      fZ = CLHEP::RandFlat::shoot(engine, fZMin, fZMax);

      if (!fControlledByEta) {
        fR = CLHEP::RandFlat::shoot(engine, fRMin, fRMax);
        fEta = std::asinh(fZ / fR);
      } else {
        fEta = CLHEP::RandFlat::shoot(engine, fEtaMin, fEtaMax);
        fR = fZ / std::sinh(fEta);
      }
    } else {
      fR = CLHEP::RandFlat::shoot(engine, fRMin, fRMax);
      fEta = CLHEP::RandFlat::shoot(engine, fEtaMin, fEtaMax);
      fZ = std::sinh(fEta) * fR;
    }

    if (fUseDeltaT) {
      fT = CLHEP::RandFlat::shoot(engine, fTMin, fTMax);
    }

    const double tmpPhi = phi;
    const double tmpR = fR;

    for (unsigned int ip = 0; ip < numParticles; ++ip) {
      if (fOverlapping) {
        fR = CLHEP::RandFlat::shoot(engine, tmpR - fDelta, tmpR + fDelta);
        phi = CLHEP::RandFlat::shoot(engine, tmpPhi - fDelta / fR, tmpPhi + fDelta / fR);
      } else {
        phi += (ip == 0 ? 0. : fDelta / fR);
      }

      double fVar = 0.;
      if (numParticles > 1 and fMaxVarSpread) {
        fVar = fVarMin + ip * (fVarMax - fVarMin) / (numParticles - 1);
      } else if (fLogSpacedVar) {
        const double fVarLog = CLHEP::RandFlat::shoot(engine, log_fVarMin, log_fVarMax);
        fVar = std::exp(fVarLog);
      } else {
        fVar = CLHEP::RandFlat::shoot(engine, fVarMin, fVarMax);
      }

      const auto partIdx = static_cast<std::size_t>(CLHEP::RandFlat::shoot(engine, 0, fPartIDs.size()));
      const int partID = fPartIDs[partIdx];

      const HepPDT::ParticleData* pData = fPDGTable->particle(HepPDT::ParticleID(std::abs(partID)));
      if (!pData) {
        throw cms::Exception("CloseByParticleGunProducer") << " Particle ID " << partID << " not found in PDG table";
      }

      const double mass = pData->mass().value();

      double mom = 0.;
      double px = 0.;
      double py = 0.;
      double pz = 0.;
      double energy = 0.;

      if (!fFlatPtGeneration) {
        const double mom2 = fVar * fVar - mass * mass;
        if (mom2 > 0.) {
          mom = std::sqrt(mom2);
        }
        pz = mom;
        energy = fVar;
      } else {
        const double theta = 2. * std::atan(std::exp(-fEta));
        mom = fVar / std::sin(theta);
        px = fVar * std::cos(phi);
        py = fVar * std::sin(phi);
        pz = mom * std::cos(theta);
        energy = std::sqrt(mom * mom + mass * mass);
      }

      const double x = fR * std::cos(phi);
      const double y = fR * std::sin(phi);
      const double r3d = std::hypot(fR, fZ);
      const double rhoXY = std::hypot(x, y);

      if (fPointing) {
        math::XYZVector direction(x, y, fZ);
        math::XYZVector momentum = direction.unit() * mom;
        px = momentum.x();
        py = momentum.y();
        pz = momentum.z();
      }

      HepMC::FourVector p(px, py, pz, energy);

      constexpr double kGeVToCmFactor = 1.e5;

      double pathLength = 0.;
      const double pt = std::hypot(p.px(), p.py());
      const double pabs = std::hypot(pt, p.pz());
      const double speed = pabs / p.e() * c_light / CLHEP::cm;

      if (pData->charge() != 0. and bz != 0. and pt > 0.) {
        // Radius [cm] = pT[GeV/c] * 1e5 / (|q| * c[mm/ns] * |B|[T])
        const double radius = pt * kGeVToCmFactor / (std::abs(pData->charge()) * c_light * std::abs(bz));
        const double arg = std::clamp(rhoXY / (2. * radius), 0., 1.);
        const double arc = 2. * std::asin(arg) * radius;
        pathLength = std::hypot(arc, fZ);
      } else {
        pathLength = r3d;
      }

      // If not pointing, the time interpretation is not very meaningful: keep the straight-line estimate.
      const double pathTime = fPointing ? pathLength / speed : r3d / speed;
      const double timeOffset = fOffsetFirst + (pathTime + ip * fT) * CLHEP::ns * c_light;

      HepMC::GenVertex* vtx =
          new HepMC::GenVertex(HepMC::FourVector(x * CLHEP::cm, y * CLHEP::cm, fZ * CLHEP::cm, timeOffset));

      HepMC::GenParticle* part = new HepMC::GenParticle(p, partID, 1);
      part->suggest_barcode(barcode);
      ++barcode;

      vtx->add_particle_out(part);

      if (fVerbosity > 0) {
        vtx->print();
        part->print();
      }

      fEvt->add_vertex(vtx);
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
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::CloseByParticleGunProducer;
DEFINE_FWK_MODULE(CloseByParticleGunProducer);
