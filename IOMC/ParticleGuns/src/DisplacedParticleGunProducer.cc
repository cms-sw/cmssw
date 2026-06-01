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
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace edm {

  namespace {

    // Hard-coded HGCAL CE-E back surface of layer 25/26
    // or, equivalently, of front face of CE-E backplate absorber 1
    // values in centimeters
    constexpr double kCeeBackZ = 362.18;
    constexpr double kCeeBackRMin = 31.36;
    constexpr double kCeeBackRMax = 164.67;

    // Sample r uniformly in area between [rmin, rmax] (uniform point density)
    double shootUniformDensity(CLHEP::HepRandomEngine* eng, double rmin, double rmax) {
      const double r2 = CLHEP::RandFlat::shoot(eng, rmin * rmin, rmax * rmax);
      return std::sqrt(r2);
    }

    // Sample r uniformly between [rmin, rmax]
    double shootUniformR(CLHEP::HepRandomEngine* eng, double rmin, double rmax) {
      return CLHEP::RandFlat::shoot(eng, rmin, rmax);
    }

    // Ensure the particle hits HGCAL's CE-E back surface within [rMin; rMax]
    bool hitsZPlaneWithinR(double x0,
                           double y0,
                           double z0,  // vertex coordinates
                           double px,
                           double py,
                           double pz,  // particle momentum
                           double zPlane,
                           double rMin,
                           double rMax,
                           int verbose) {
      const double t = (zPlane - z0) / pz;
      if (t <= 0.0) {
        return false;
      }

      // project (x, y) into the plane assuming straight trajectories
      const double xHit = x0 + t * px;
      const double yHit = y0 + t * py;
      const double rHit = std::hypot(xHit, yHit);

      if (verbose > 0) {
        std::cout << "hitsZPlaneWithin " << " | return=" << static_cast<int>(rHit >= rMin && rHit <= rMax)
                  << " | rHit=" << rHit << ", rMin=" << rMin << ", rMax=" << rMax << ", t=" << t
                  << ", zPlane=" << zPlane << ", z0=" << z0 << ", pz=" << pz << std::endl;
      }

      return (rHit >= rMin && rHit <= rMax);
    }

    // ensures theta is never too close to zero, which makes the computation of pz unstable
    double pickSensibleTheta(CLHEP::HepRandomEngine* eng, double amin, double amax) {
      double theta = 0.;
      while (std::abs(theta) < 1e-6) {
        theta = CLHEP::RandFlat::shoot(eng, amin, amax);
      }
      return theta;
    }

    std::tuple<double, double, double> computeMomentum(double pt, double theta, double phi) {
      double px = pt * std::cos(phi);
      double py = pt * std::sin(phi);
      double pz = 0.;
      if (std::abs(theta) < 1e-6) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Theta is too close to zero: " << theta << ". Unstable pz.";
      }

      pz = pt / std::abs(std::tan(theta));

      // shoot the particle along the same plane but in the negative theta direction
      // reminder: theta is measured from the x axis in counter-clockwise fashion
      if (theta < 0.) {
        px = -px;
        py = -py;
      }

      return {px, py, pz};
    }

  }  // namespace

  class DisplacedParticleGunProducer : public edm::global::EDProducer<> {
  public:
    explicit DisplacedParticleGunProducer(const ParameterSet&);
    ~DisplacedParticleGunProducer() override = default;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const override;

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
    int fPartID;
    bool fUniformDensityInR = false;
    unsigned int fMaxTries = 1000;

    // If true: derive theta range from hard-coded HGCAL CE-E back surface envelope
    //   (with R in [RMinBackSurfaceHGCAL, RMaxBackSurfaceHGCAL]) and vertex rho
    // If false: sample theta uniformly in [MinTheta, MaxTheta]
    bool fPointingToHGCAL = true;
    bool fRestrictRInZPlaneAtZero = true;
    double fRMinBackSurfaceHGCAL = kCeeBackRMin;
    double fRMaxBackSurfaceHGCAL = kCeeBackRMax;
    double fThetaMin = 0.;
    double fThetaMax = 0.;
    std::optional<double> fRMinAtZero = std::nullopt;
    std::optional<double> fRMaxAtZero = std::nullopt;

    const ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> fPDGTableToken;
    int fVerbosity = 0;
  };

  DisplacedParticleGunProducer::DisplacedParticleGunProducer(const ParameterSet& pset)
      : fPDGTableToken(esConsumes<>()) {
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
    fThetaMin = pgun.getParameter<double>("MinTheta");
    fThetaMax = pgun.getParameter<double>("MaxTheta");
    fPhiVtxMin = pgun.getParameter<double>("MinVtxPhi");
    fPhiVtxMax = pgun.getParameter<double>("MaxVtxPhi");
    fRMin = pgun.getParameter<double>("RMin");
    fRMax = pgun.getParameter<double>("RMax");
    fZVtx = pgun.getParameter<double>("ZVtx");
    fNParticles = pgun.getParameter<int>("NParticles");
    fPartID = pgun.getParameter<int>("PartID");
    fUniformDensityInR = pgun.getParameter<bool>("UniformDensityInR");
    fMaxTries = pgun.getParameter<unsigned int>("MaxTries");
    fVerbosity = pset.getUntrackedParameter<int>("Verbosity");
    fPointingToHGCAL = pgun.getParameter<bool>("PointingToHGCAL");
    fRestrictRInZPlaneAtZero = pgun.getParameter<bool>("RestrictRInZPlaneAtZero");

    if (fRestrictRInZPlaneAtZero && !fPointingToHGCAL) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Currently RestrictRInZPlaneAtZero only works if PointingToHGCAL is active.";
    }

    if (fPointingToHGCAL) {
      fRMinBackSurfaceHGCAL = pgun.getParameter<double>("RMinBackSurfaceHGCAL");
      fRMaxBackSurfaceHGCAL = pgun.getParameter<double>("RMaxBackSurfaceHGCAL");
    }

    if (fRestrictRInZPlaneAtZero) {
      fRMinAtZero = pgun.getParameter<double>("RMinAtZero");
      fRMaxAtZero = pgun.getParameter<double>("RMaxAtZero");
      if (fRMaxAtZero <= fRMinAtZero) {
        throw cms::Exception("DisplacedParticleGunProducer") << "Please fix RMaxAtZero/RMinAtZero";
      }
      if (fRMinAtZero < 0.) {
        throw cms::Exception("DisplacedParticleGunProducer") << "RMinAtZero must be positive.";
      }
    }

    if (fPtMax <= fPtMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinPt/MaxPt";
    }
    if (fPhiMax <= fPhiMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinPhi/MaxPhi";
    }
    if (fThetaMax <= fThetaMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please ensure MinTheta <= MaxTheta.";
    }
    if (fPhiVtxMax <= fPhiVtxMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix MinVtxPhi/MaxVtxPhi";
    }
    if (fRMax <= fRMin) {
      throw cms::Exception("DisplacedParticleGunProducer") << "Please fix RMin/RMax";
    }
    if (fRestrictRInZPlaneAtZero && fPointingToHGCAL && fRMax > fRMaxAtZero && fRMax > fRMaxBackSurfaceHGCAL) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "There are values of R at z=" << fZVtx << "cm for which an intersection for R in [" << *fRMinAtZero << "; "
          << *fRMaxAtZero << "]cm at z=0cm is impossible. Please update your configuration.";
    }
    if (fRMin < 0) {
      throw cms::Exception("DisplacedParticleGunProducer") << "RMin must be positive.";
    }
    if (fMaxTries == 0) {
      throw cms::Exception("DisplacedParticleGunProducer") << "MaxTries must be > 0";
    }
    if (fRMaxBackSurfaceHGCAL <= fRMinBackSurfaceHGCAL) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Please ensure RMaxBackSurfaceHGCAL > RMinBackSurfaceHGCAL.";
    }
    if (fRMaxBackSurfaceHGCAL > kCeeBackRMax || fRMinBackSurfaceHGCAL < kCeeBackRMin) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Please ensure RMaxBackSurfaceHGCAL <= kCeeBackRMax and RMinBackSurfaceHGCAL >= kCeeBackRMin.";
    }

    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
  }

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
    pgun.add<int>("PartID", 22);

    pgun.add<bool>("UniformDensityInR", false);

    pgun.add<unsigned int>("MaxTries", 1000u);

    // A particle shot at the extremities of the HGCAL surface will not traverse a substantial fraction of the detector.
    // We use RMinBackSurfaceHGCAL and RMaxBackSurfaceHGCAL to expose only a given region of the HGCAL surface
    // The default arguments correspond to the inner third of HGCAL's surface (R in ~[58.79, 91.58]cm).
    // At (R=200,z=0)cm, an uncharged particle pointing to R=58.79cm at the HGCAL surface (the most extreme case) exits the calorimeter at the
    //  back face of the CE-E, crossing all its layers.
    // For R>200cm there is no guarantee all CE-E layers will be crossed, so a tighter R range might be needed.
    // For z>0cm the angles will become more extreme, so a tighter R range might be needed.
    // The above reasoning breaks for charged particles, since the bending under the magnetic filed can enormously extend the particle's reach.
    pgun.add<bool>("PointingToHGCAL", true);
    pgun.add<double>("RMinBackSurfaceHGCAL", 58.79);
    pgun.add<double>("RMaxBackSurfaceHGCAL", 91.58);
    pgun.add<double>("MinTheta", -std::numbers::pi / 2 + 1e-6);
    pgun.add<double>("MaxTheta", std::numbers::pi / 2 - 1e-6);

    pgun.add<bool>("RestrictRInZPlaneAtZero", true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("RMinAtZero", 0., true), true);
    pgun.addOptionalNode(edm::ParameterDescription<double>("RMaxAtZero", 150., true), true);

    desc.add<edm::ParameterSetDescription>("PGunParameters", pgun);

    desc.addUntracked<int>("Verbosity", 0);
    desc.addUntracked<unsigned int>("firstRun", 1);

    descriptions.add("DisplacedParticleGunProducer", desc);
  }

  void DisplacedParticleGunProducer::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const {
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    if (fVerbosity > 0) {
      LogDebug("DisplacedParticleGunProducer")
          << " DisplacedParticleGunProducer : Begin New Event Generation" << std::endl;
    }

    HepMC::GenEvent* fEvt = new HepMC::GenEvent();

    if (fPointingToHGCAL) {
      if (kCeeBackZ <= fZVtx) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Invalid hard-coded HGCAL surface envelope: "
            << "kCeeBackZ = " << kCeeBackZ << "cm, fZVtx = " << fZVtx << " (check ZVtx).";
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

      auto const& pdgTable = es.getData(fPDGTableToken);
      const HepPDT::ParticleData* pData = pdgTable.particle(HepPDT::ParticleID(std::abs(fPartID)));
      if (!pData) {
        throw cms::Exception("DisplacedParticleGunProducer") << "Particle ID " << fPartID << " not found in PDG table";
      }
      const double mass = pData->mass().value();

      if (pData->charge() != 0 && fPointingToHGCAL) {
        throw cms::Exception("DisplacedParticleGunProducer") << "The logic that points particles to HGCAL's CE-E back "
                                                                "face assumes that particles move in straight lines.";
      }

      double theta = 0., px = 0., py = 0., pz = 0.;
      double phi = phiVtx; /* the particle's direction has the same phi as its vertex, ie.,
							  it moves on a 2D plane parallel to the z axis */
      const double pt = CLHEP::RandFlat::shoot(engine, fPtMin, fPtMax);
      if (fPointingToHGCAL) {
        bool accepted = false;
        for (unsigned int itry = 0; itry < fMaxTries; ++itry) {
          theta = pickSensibleTheta(engine, fThetaMin, fThetaMax);
          std::tie(px, py, pz) = computeMomentum(pt, theta, phi);
          if (edm::isNotFinite(pz) || pz <= 0.0) {
            continue;  // must go towards +z plane
          }
          if (fVerbosity > 0) {
            std::cout << "phiVtx=" << phiVtx << ", RVtx=" << RVtx << ", pT=" << pt << ", theta=" << theta
                      << ", phi=" << phi << std::endl;
          }

          bool checkBackSurface = hitsZPlaneWithinR(
              xVtx, yVtx, fZVtx, px, py, pz, kCeeBackZ, fRMinBackSurfaceHGCAL, fRMaxBackSurfaceHGCAL, fVerbosity);
          bool checkZero =
              fRestrictRInZPlaneAtZero &&
              hitsZPlaneWithinR(
                  xVtx, yVtx, fZVtx, -px, -py, -pz, 0., fRMinAtZero.value(), fRMaxAtZero.value(), fVerbosity);
          if (checkBackSurface && checkZero) {
            accepted = true;
            break;
          }
        }
        if (!accepted) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Failed to generate a particle intersecting HGCAL CE-E back surface after MaxTries=" << fMaxTries
              << ". Vertex located at: (R=" << RVtx << "cm, phiVtx=" << phiVtx << ", z=" << fZVtx
              << "cm). HGCAL band located at R in [" << fRMinBackSurfaceHGCAL << "; " << fRMaxBackSurfaceHGCAL
              << "]cm at z=" << kCeeBackZ << "cm.";
        }
      } else {  // if (!fPointingToHGCAL)
        theta = pickSensibleTheta(engine, fThetaMin, fThetaMax);
        phi = CLHEP::RandFlat::shoot(engine, fPhiMin, fPhiMax);
        std::tie(px, py, pz) = computeMomentum(pt, theta, phi);
      }

      const double p2 = px * px + py * py + pz * pz;
      const double energy = std::sqrt(p2 + mass * mass);

      HepMC::FourVector p(px, py, pz, energy);

      HepMC::GenVertex* vtx =
          new HepMC::GenVertex(HepMC::FourVector(xVtx * CLHEP::cm, yVtx * CLHEP::cm, fZVtx * CLHEP::cm, 0.0));

      HepMC::GenParticle* part = new HepMC::GenParticle(p, fPartID, 1);
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
      std::cout << " DisplacedParticleGunProducer : Event Generation Done. " << std::endl;
    }
  }

}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::DisplacedParticleGunProducer;
DEFINE_FWK_MODULE(DisplacedParticleGunProducer);
