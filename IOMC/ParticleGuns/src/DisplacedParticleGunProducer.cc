#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>
#include <string>
#include <tuple>
#include <optional>

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Units/SystemOfUnits.h>

#include "HepMC/GenEvent.h"

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace edm {

  namespace {

    // --- Hard-coded HGCAL front surface ---
    constexpr double kHGCalZ = 296.50;
    constexpr double kHGCalRMin = 26.00;
    constexpr double kHGCalRMax = 124.37;

    // Sample r uniformly in area between [rmin, rmax] (uniform point density)
    double shootUniformDensity(CLHEP::HepRandomEngine* eng, double rmin, double rmax) {
      const double r2 = CLHEP::RandFlat::shoot(eng, rmin * rmin, rmax * rmax);
      return std::sqrt(r2);
    }

    // Sample r uniformly between [rmin, rmax]
    double shootUniformR(CLHEP::HepRandomEngine* eng, double rmin, double rmax) {
      return CLHEP::RandFlat::shoot(eng, rmin, rmax);
    }

    // Ensure the particle hits HGCAL's front surface within [rMin; rMax]
    bool hitsZPlaneWithinR(double x0,
                           double y0,
                           double z0,  // vertex coordinates
                           double px,
                           double py,
                           double pz,  // particle momentum
                           double zPlane,
                           double rMin,
                           double rMax) {  // hard-coded HGCAL geometry
      // Must move towards the plane: require (zPlane - z0) / pz > 0
      const double dz = zPlane - z0;
      if (pz < 1e-6) {
        return false;
      }
      const double t = dz / pz;
      if (t <= 0.0) {
        return false;
      }

      const double xHit = x0 + t * px;
      const double yHit = y0 + t * py;
      const double rHit = std::hypot(xHit, yHit);

      return (rHit >= rMin && rHit <= rMax);
    }

    // Compute the theta range (in radians) that intersects a plane at z=zFront,
    // within radial bounds [rMin, rMax], from a vertex at transverse radius R0 and z0.
    std::pair<double, double> thetaRangeToPointToHGCALSurface(
        double R, double z, double zHGCAL, double rminHGCAL, double rmaxHGCAL) {
      const double dz = zHGCAL - z;
      if (dz <= 0. or rminHGCAL > rmaxHGCAL) {
        return {0., 0.};
      }

      double thetaMin = std::atan((rminHGCAL - R) / dz);
      double thetaMax = std::atan((rmaxHGCAL - R) / dz);

      // For +z endcap, theta should be in (0, pi/2)
      // clamping removes instabilities for pz
      thetaMin = std::clamp(thetaMin, 1e-6, 0.5 * std::numbers::pi - 1e-6);
      thetaMax = std::clamp(thetaMax, 1e-6, 0.5 * std::numbers::pi - 1e-6);

      if (thetaMax <= thetaMin) {
        return {0., 0.};
      }
      return {thetaMin, thetaMax};
    }

    std::tuple<double, double, double> computeMomentum(double pt, double theta, double phi) {
      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = 0.;
      if (theta < 1e-6) {
        throw cms::Exception("DisplacedParticleGunProducer") << "Theta is too small. Unstable pz.";
      } else {
        pz = pt / std::tan(theta);
      }
      return {px, py, pz};
    }

  }  // namespace

  class DisplacedParticleGunProducer : public one::EDProducer<one::WatchRuns, EndRunProducer> {
  public:
    explicit DisplacedParticleGunProducer(const ParameterSet&);
    ~DisplacedParticleGunProducer() override = default;

    void beginRun(const edm::Run& r, const edm::EventSetup&) override;
    void endRun(edm::Run const& r, const edm::EventSetup&) override;
    void endRunProduce(edm::Run& r, const edm::EventSetup&) override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(Event& e, const EventSetup& es) override;

    double fPtMin = 0.;
    double fPtMax = 0.;
    double fPhiMin = 0.;
    double fPhiMax = 0.;
    double fRMin = 0.;
    double fRMax = 0.;
    double fPhiVtxMin = 0.;
    double fPhiVtxMax = 0.;
    double fZVtx = 0.;
    int fNParticles = 1;
    std::vector<int> fPartIDs;
    bool fUniformDensityInR = false;
    unsigned int fMaxTries = 1000;

    // If true: derive theta range from hard-coded HGCAL front surface envelope
    //   (with R in [RminFrontSurfaceHGCAL, RmaxFrontSurfaceHGCAL]) and vertex rho
    // If false: sample theta uniformly in [MinTheta, MaxTheta]
    bool fPointingToHGCAL = true;
    double fRminFrontSurfaceHGCAL = kHGCalRMin;
    double fRmaxFrontSurfaceHGCAL = kHGCalRMax;
    std::optional<double> fThetaMin = 0.;
    std::optional<double> fThetaMax = 0.;

    ESHandle<HepPDT::ParticleDataTable> fPDGTable;
    const ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> fPDGTableToken;
    HepMC::GenEvent* fEvt;
    int fVerbosity = 0;
  };

  DisplacedParticleGunProducer::DisplacedParticleGunProducer(const ParameterSet& pset)
      : fPDGTableToken(esConsumes<Transition::BeginRun>()), fEvt(nullptr) {
    Service<RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
          << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
             "which appears to be absent.  Please add that service to your configuration\n"
             "or remove the modules that require it.";
    }

    const auto pgun = pset.getParameter<ParameterSet>("PGunParameters");

    fPtMin = pgun.getParameter<double>("MinPt");
    fPtMax = pgun.getParameter<double>("MaxPt");
    fPhiMin = pgun.getParameter<double>("MinPhi");
    fPhiMax = pgun.getParameter<double>("MaxPhi");
    fPhiVtxMin = pgun.getParameter<double>("MinVtxPhi");
    fPhiVtxMax = pgun.getParameter<double>("MaxVtxPhi");
    fRMin = pgun.getParameter<double>("RMin");
    fRMax = pgun.getParameter<double>("RMax");
    fZVtx = pgun.getParameter<double>("ZVtx");
    fNParticles = pgun.getParameter<int>("NParticles");
    fPartIDs = pgun.getParameter<std::vector<int>>("PartID");
    fUniformDensityInR = pgun.getParameter<bool>("UniformDensityInR");
    fMaxTries = pgun.getParameter<unsigned int>("MaxTries");
    fPointingToHGCAL = pgun.getParameter<bool>("PointingToHGCAL");

    if (!fPointingToHGCAL) {
      fThetaMin = pgun.getParameter<double>("MinTheta");
      fThetaMax = pgun.getParameter<double>("MaxTheta");
      if (*fThetaMax <= *fThetaMin) {
        throw cms::Exception("DisplacedParticleGunProducer") << "Please ensure MinTheta <= MaxTheta.";
      }
    } else {
      fRminFrontSurfaceHGCAL = pgun.getParameter<double>("RminFrontSurfaceHGCAL");
      fRmaxFrontSurfaceHGCAL = pgun.getParameter<double>("RmaxFrontSurfaceHGCAL");
      fThetaMin = std::nullopt;
      fThetaMax = std::nullopt;
    }

    if (fPtMax <= fPtMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinPt/MaxPt";
    }
    if (fPhiMax <= fPhiMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinPhi/MaxPhi";
    }
    if (fPhiVtxMax <= fPhiVtxMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinVtxPhi/MaxVtxPhi";
    }
    if (fRMax <= fRMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix RMin/RMax";
    }
    if (fPartIDs.empty()) {
      throw cms::Exception("DisplacedParticleGunProducer") << "PartID must be non-empty";
    }
    if (fMaxTries == 0) {
      throw cms::Exception("DisplacedParticleGunProducer") << "MaxTries must be > 0";
    }
	if (fRmaxFrontSurfaceHGCAL <= fRminFrontSurfaceHGCAL) {
	  throw cms::Exception("DisplacedParticleGunProducer")
		<< "Please ensure RmaxFrontSurfaceHGCAL > RminFrontSurfaceHGCAL.";
	}
	if (fRmaxFrontSurfaceHGCAL > kHGCalRMax || fRminFrontSurfaceHGCAL < kHGCalRMin) {
	  throw cms::Exception("DisplacedParticleGunProducer")
		<< "Please ensure RmaxFrontSurfaceHGCAL <= kHGCalRMax and RminFrontSurfaceHGCAL >= kHGCalRMin.";
	}
	  
    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
  }

  void DisplacedParticleGunProducer::beginRun(const edm::Run& r, const EventSetup& es) {
    fPDGTable = es.getHandle(fPDGTableToken);
    return;
  }

  void DisplacedParticleGunProducer::endRun(const Run& run, const EventSetup& es) {}

  void DisplacedParticleGunProducer::endRunProduce(Run& run, const EventSetup& es) {}

  void DisplacedParticleGunProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("AddAntiParticle", false);

    edm::ParameterSetDescription pgun;

    // particle direction
    pgun.add<double>("MinPt", 5.);
    pgun.add<double>("MaxPt", 100.);
    pgun.add<double>("MinPhi", -std::numbers::pi);
    pgun.add<double>("MaxPhi", +std::numbers::pi);

    // vertex displacement (cm)
    pgun.add<double>("RMin", 0.);
    pgun.add<double>("RMax", 10.);
    pgun.add<double>("MinVtxPhi", 0.);
    pgun.add<double>("MaxVtxPhi", 2 * std::numbers::pi);
    pgun.add<double>("ZVtx", 0.);

    pgun.add<int>("NParticles", 1);
    pgun.add<std::vector<int>>("PartID", {22});

    pgun.add<bool>("UniformDensityInR", false);

    pgun.add<unsigned int>("MaxTries", 1000u);

    // A particle shot at the extremities of the HGCAL surface will not traverse a substantial fraction of the detector.
    // We use RminFrontSurfaceHGCAL and RmaxFrontSurfaceHGCAL to expose only a given region of the HGCAL surface
    // THe default arguments correspond to the inner third of HGCAL's surface (R in ~[58.79, 91.58]cm).
    // At (R=200,z=0)cm, an uncharged particle pointing to R=58.79cm at the HGCAL surface (the most extreme case) exits the calorimeter at the
    //  back face of the CE-E, crossing all its layers.
    // For R>200cm there is no guarantee all CE-E layers will be crossed, so a tighter R range might be needed.
    // For z>0cm the angles will become more extreme, so a tighter R range might be needed.
    // The above reasoning breaks for charged particles, since the bending under the magnetic filed can enormously extend the particle's reach.
    pgun.add<bool>("PointingToHGCAL", true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("RminFrontSurfaceHGCAL", 58.79, true), true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("RmaxFrontSurfaceHGCAL", 91.58, true), true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("MinTheta", 0.2, true), true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("MaxTheta", 1.2, true), true);

    desc.add<edm::ParameterSetDescription>("PGunParameters", pgun);

    desc.addUntracked<int>("Verbosity", 0);
    desc.addUntracked<unsigned int>("firstRun", 1);
    desc.add<std::string>("psethack",
                          "displaced gun with theta, optionally pointing to hard-coded HGCAL front surface");

    descriptions.add("DisplacedParticleGunProducer", desc);
  }

  void DisplacedParticleGunProducer::produce(Event& e, const EventSetup& /*es*/) {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    if (fVerbosity > 0) {
      LogDebug("DisplacedParticleGunProducer")
          << " DisplacedParticleGunProducer : Begin New Event Generation" << std::endl;
    }

    fEvt = new HepMC::GenEvent();

    const double zFront = kHGCalZ;
    const double rMinFace = fRminFrontSurfaceHGCAL;
    const double rMaxFace = fRmaxFrontSurfaceHGCAL;

    if (fPointingToHGCAL) {
      if (zFront <= fZVtx) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Invalid hard-coded HGCAL surface envelope: "
            << "zFront = " << zFront << "cm, fZVtx = " << fZVtx << " (check ZVtx).";
      }
    }

    int barcode = 1;

    for (int ip = 0; ip < fNParticles; ++ip) {
      // --- Sample displaced vertex in transverse annulus (z fixed) ---
      const double RVtx =
          fUniformDensityInR ? shootUniformDensity(engine, fRMin, fRMax) : shootUniformR(engine, fRMin, fRMax);
      const double phiVtx = CLHEP::RandFlat::shoot(engine, fPhiVtxMin, fPhiVtxMax);
      const double xVtx = RVtx * std::cos(phiVtx);
      const double yVtx = RVtx * std::sin(phiVtx);
      const double zVtx = fZVtx;

      const auto idx = static_cast<std::size_t>(CLHEP::RandFlat::shoot(engine, 0, fPartIDs.size()));
      const int partID = fPartIDs[idx];

      const HepPDT::ParticleData* pData = fPDGTable->particle(HepPDT::ParticleID(std::abs(partID)));
      if (!pData) {
        throw cms::Exception("DisplacedParticleGunProducer") << "Particle ID " << partID << " not found in PDG table";
      }
      const double mass = pData->mass().value();

      if (pData->charge() != 0 && fPointingToHGCAL) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "The logic that points particles to HGCAL's front face assumes that particles move in straight lines.";
      }

      double theta = 0., phi = 0., px = 0., py = 0., pz = 0.;
      const double pt = CLHEP::RandFlat::shoot(engine, fPtMin, fPtMax);
      if (fPointingToHGCAL) {
        const double RVtx0 = std::hypot(xVtx, yVtx);
        auto [thetaMin, thetaMax] = thetaRangeToPointToHGCALSurface(RVtx0, zVtx, zFront, rMinFace, rMaxFace);
        if (!(thetaMax >= thetaMin)) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "No valid theta window for this vertex at RVtx=" << RVtx0 << "cm within HGCAL's front surface.";
        }

        bool accepted = false;
        for (unsigned int itry = 0; itry < fMaxTries; ++itry) {
          theta = CLHEP::RandFlat::shoot(engine, thetaMin, thetaMax);
          phi = CLHEP::RandFlat::shoot(engine, fPhiMin, fPhiMax);
          std::tie(px, py, pz) = computeMomentum(pt, theta, phi);
          if (edm::isNotFinite(pz) || pz <= 0.0) {
            continue;  // must go towards +z plane
          }

          if (hitsZPlaneWithinR(
                  xVtx, yVtx, zVtx, px, py, pz, zFront, fRminFrontSurfaceHGCAL, fRmaxFrontSurfaceHGCAL)) {
            accepted = true;
            break;
          }
        }
        if (!accepted) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Failed to generate a particle intersecting HGCAL front surface after MaxTries=" << fMaxTries
              << ". Vertex: (R=" << RVtx << "cm, phiVtx=" << phiVtx << ", z=" << zVtx << "cm). HGCAL band: ["
              << fRminFrontSurfaceHGCAL << "," << fRmaxFrontSurfaceHGCAL << "]cm at z=" << zFront << "cm.";
        }
      } else {  // if (fPointingToHGCAL)
        theta = CLHEP::RandFlat::shoot(engine, fThetaMin.value(), fThetaMax.value());
        phi = CLHEP::RandFlat::shoot(engine, fPhiMin, fPhiMax);
        std::tie(px, py, pz) = computeMomentum(pt, theta, phi);
      }

      const double p2 = px * px + py * py + pz * pz;
      const double energy = std::sqrt(p2 + mass * mass);

      HepMC::FourVector p(px, py, pz, energy);

      HepMC::GenVertex* vtx =
          new HepMC::GenVertex(HepMC::FourVector(xVtx * CLHEP::cm, yVtx * CLHEP::cm, zVtx * CLHEP::cm, 0.0));

      HepMC::GenParticle* part = new HepMC::GenParticle(p, partID, 1);
      part->suggest_barcode(barcode++);

      vtx->add_particle_out(part);
      fEvt->add_vertex(vtx);

      if (fVerbosity > 0) {
        vtx->print();
        part->print();
      }
    }

    fEvt->set_event_number(e.id().event());
    fEvt->set_signal_process_id(20);

    if (fVerbosity > 0) {
      fEvt->print();
    }

    auto bProduct = std::make_unique<HepMCProduct>();
    bProduct->addHepMCData(fEvt);
    e.put(std::move(bProduct), "unsmeared");

    auto genEventInfo = std::make_unique<GenEventInfoProduct>(fEvt);
    e.put(std::move(genEventInfo));

    if (fVerbosity > 0) {
      LogDebug("DisplacedParticleGunProducer") << " DisplacedParticleGunProducer : Event Generation Done " << std::endl;
    }
  }

}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::DisplacedParticleGunProducer;
DEFINE_FWK_MODULE(DisplacedParticleGunProducer);
