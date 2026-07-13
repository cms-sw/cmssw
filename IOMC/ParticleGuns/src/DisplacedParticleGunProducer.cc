#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace edm {

  namespace {

    enum class MagnitudeVariable { kEnergy, kPt };
    enum class SampleAt { kOrigin, kProduction };
    enum class RadialDistribution { kUniformArea, kUniformRadius };

    MagnitudeVariable readMagnitudeVariable(const std::string& value) {
      if (value == "energy") {
        return MagnitudeVariable::kEnergy;
      }
      if (value == "pt") {
        return MagnitudeVariable::kPt;
      }
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Momentum.Magnitude.Variable must be either 'energy' or 'pt', but is '" << value << "'.";
    }

    SampleAt readSampleAt(const std::string& value) {
      if (value == "origin") {
        return SampleAt::kOrigin;
      }
      if (value == "production") {
        return SampleAt::kProduction;
      }
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Geometry.SampleAt must be either 'origin' or 'production', but is '" << value << "'.";
    }

    RadialDistribution readRadialDistribution(const std::string& value) {
      if (value == "uniformArea") {
        return RadialDistribution::kUniformArea;
      }
      if (value == "uniformRadius") {
        return RadialDistribution::kUniformRadius;
      }
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Geometry.RadialDistribution must be either 'uniformArea' or 'uniformRadius', but is '" << value << "'.";
    }

    struct MagnitudeParameters {
      explicit MagnitudeParameters(const ParameterSet& pset)
          : variable(readMagnitudeVariable(pset.getParameter<std::string>("Variable"))),
            min(pset.getParameter<double>("Min")),
            max(pset.getParameter<double>("Max")) {
        if (max <= min) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Please fix Momentum.Magnitude.Min/Max.";
        }
        if (min <= 0.) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Momentum.Magnitude.Min must be positive.";
        }
      }

      MagnitudeVariable variable;
      double min;
      double max;
    };

    struct DirectionParameters {
      explicit DirectionParameters(const ParameterSet& pset)
          : thetaMin(pset.getParameter<double>("ThetaMin")),
            thetaMax(pset.getParameter<double>("ThetaMax")),
            phiMin(pset.getParameter<double>("PhiMin")),
            phiMax(pset.getParameter<double>("PhiMax")) {
        if (thetaMax <= thetaMin) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Please fix Momentum.Direction.ThetaMin/ThetaMax.";
        }
        if (phiMax <= phiMin) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Please fix Momentum.Direction.PhiMin/PhiMax.";
        }
        if (thetaMin <= 0. || thetaMax >= std::numbers::pi / 2.) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Momentum.Direction theta bounds must lie inside (0, pi/2).";
        }
      }

      double thetaMin;
      double thetaMax;
      double phiMin;
      double phiMax;
    };

    struct MomentumParameters {
      explicit MomentumParameters(const ParameterSet& pset)
          : magnitude(pset.getParameter<ParameterSet>("Magnitude")),
            direction(pset.getParameter<ParameterSet>("Direction")) {}

      MagnitudeParameters magnitude;
      DirectionParameters direction;
    };

    struct PlaneParameters {
      PlaneParameters(const ParameterSet& pset, double planeZ, const char* name)
          : z(planeZ),
            rMin(pset.getParameter<double>("RMin")),
            rMax(pset.getParameter<double>("RMax")),
            phiMin(pset.getParameter<double>("PhiMin")),
            phiMax(pset.getParameter<double>("PhiMax")) {
        if (rMax <= rMin) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Please fix Geometry." << name << ".RMin/RMax.";
        }
        if (rMin < 0.) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Geometry." << name << ".RMin must be nonnegative.";
        }
        if (phiMax <= phiMin) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Please fix Geometry." << name << ".PhiMin/PhiMax.";
        }
      }

      PlaneParameters(const ParameterSet& pset, const char* name)
          : PlaneParameters(pset, pset.getParameter<double>("Z"), name) {}

      double z;
      double rMin;
      double rMax;
      double phiMin;
      double phiMax;
    };

    std::optional<PlaneParameters> readTarget(const ParameterSet& geometry) {
      if (!geometry.existsAs<ParameterSet>("Target")) {
        return std::nullopt;
      }
      return PlaneParameters(geometry.getParameter<ParameterSet>("Target"), "Target");
    }

    struct GeometryParameters {
      explicit GeometryParameters(const ParameterSet& pset)
          : sampleAt(readSampleAt(pset.getParameter<std::string>("SampleAt"))),
            radialDistribution(readRadialDistribution(pset.getParameter<std::string>("RadialDistribution"))),
            origin(pset.getParameter<ParameterSet>("Origin"), 0., "Origin"),
            production(pset.getParameter<ParameterSet>("Production"), "Production"),
            target(readTarget(pset)) {
        if (production.z <= origin.z) {
          throw cms::Exception("DisplacedParticleGunProducer") << "Geometry.Production.Z must be greater than zero.";
        }
        if (target && target->z <= production.z) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Geometry.Target.Z must be greater than Geometry.Production.Z.";
        }
        if (target) {
          const double fraction = production.z / target->z;
          const double largestReachableProductionRadius = (1. - fraction) * origin.rMax + fraction * target->rMax;
          if (production.rMin > largestReachableProductionRadius) {
            throw cms::Exception("DisplacedParticleGunProducer")
                << "Geometry.Production radial range is unreachable between the configured Origin and Target caps.";
          }
        }
      }

      SampleAt sampleAt;
      RadialDistribution radialDistribution;
      PlaneParameters origin;
      PlaneParameters production;
      std::optional<PlaneParameters> target;
    };

    double distanceBetweenIntervals(double firstMin, double firstMax, double secondMin, double secondMax) {
      if (firstMax < secondMin) {
        return secondMin - firstMax;
      }
      if (secondMax < firstMin) {
        return firstMin - secondMax;
      }
      return 0.;
    }

    bool capIsPairwiseReachable(const PlaneParameters& sampled,
                                const PlaneParameters& required,
                                const DirectionParameters& direction) {
      const double deltaZ = std::abs(required.z - sampled.z);
      const double displacementMin = deltaZ * std::tan(direction.thetaMin);
      const double displacementMax = deltaZ * std::tan(direction.thetaMax);
      const double reachableRMin =
          distanceBetweenIntervals(sampled.rMin, sampled.rMax, displacementMin, displacementMax);
      const double reachableRMax = sampled.rMax + displacementMax;
      return required.rMax >= reachableRMin && required.rMin <= reachableRMax;
    }

    struct ParticleGunParameters {
      explicit ParticleGunParameters(const ParameterSet& pset)
          : partId(pset.getParameter<int>("PartID")),
            nParticles(pset.getParameter<int>("NParticles")),
            momentum(pset.getParameter<ParameterSet>("Momentum")),
            geometry(pset.getParameter<ParameterSet>("Geometry")),
            maxDirectionTries(pset.getParameter<unsigned int>("MaxDirectionTries")) {
        if (nParticles <= 0) {
          throw cms::Exception("DisplacedParticleGunProducer") << "NParticles must be greater than zero.";
        }
        if (maxDirectionTries == 0) {
          throw cms::Exception("DisplacedParticleGunProducer") << "MaxDirectionTries must be greater than zero.";
        }

        const auto& sampled = geometry.sampleAt == SampleAt::kOrigin ? geometry.origin : geometry.production;
        const auto checkReachable = [&](const PlaneParameters& required, const char* name) {
          if (!capIsPairwiseReachable(sampled, required, momentum.direction)) {
            throw cms::Exception("DisplacedParticleGunProducer")
                << "Geometry." << name << " radial range is unreachable from the sampled cap within the theta range.";
          }
        };
        if (geometry.sampleAt == SampleAt::kOrigin) {
          checkReachable(geometry.production, "Production");
        } else {
          checkReachable(geometry.origin, "Origin");
        }
        if (geometry.target) {
          checkReachable(*geometry.target, "Target");
        }
      }

      int partId;
      int nParticles;
      MomentumParameters momentum;
      GeometryParameters geometry;
      unsigned int maxDirectionTries;
    };

    struct ProducerParameters {
      explicit ProducerParameters(const ParameterSet& pset)
          : particleGun(pset.getParameter<ParameterSet>("PGunParameters")),
            verbosity(pset.getUntrackedParameter<int>("Verbosity", 0)) {}

      ParticleGunParameters particleGun;
      int verbosity;
    };

    struct Interval {
      double min;
      double max;
    };

    struct Point {
      double x;
      double y;
    };

    struct FourMomentum {
      double px;
      double py;
      double pz;
      double energy;
    };

    struct ProductionVertex {
      double x;
      double y;
      double z;
      double time;
    };

    struct ResolvedParticle {
      FourMomentum momentum;
      ProductionVertex vertex;
    };

    std::vector<Interval> intersectIntervals(const std::vector<Interval>& first, const std::vector<Interval>& second) {
      std::vector<Interval> result;
      for (const auto& left : first) {
        for (const auto& right : second) {
          const double min = std::max(left.min, right.min);
          const double max = std::min(left.max, right.max);
          if (min < max) {
            result.push_back({min, max});
          }
        }
      }
      return result;
    }

    std::optional<Interval> quadraticRoots(double a, double b, double c) {
      const double discriminant = b * b - 4. * a * c;
      if (discriminant <= 0.) {
        return std::nullopt;
      }

      const double sqrtDiscriminant = std::sqrt(discriminant);
      return Interval{(-b - sqrtDiscriminant) / (2. * a), (-b + sqrtDiscriminant) / (2. * a)};
    }

    std::vector<Interval> allowedSlopesForCap(double sampledR,
                                              double deltaPhi,
                                              double deltaZ,
                                              const PlaneParameters& required) {
      const double a = deltaZ * deltaZ;
      const double b = 2. * deltaZ * sampledR * std::cos(deltaPhi);
      const double c = sampledR * sampledR;

      const auto outerRoots = quadraticRoots(a, b, c - required.rMax * required.rMax);
      if (!outerRoots) {
        return {};
      }

      std::vector<Interval> allowed{{outerRoots->min, outerRoots->max}};
      const auto innerRoots = quadraticRoots(a, b, c - required.rMin * required.rMin);
      if (innerRoots) {
        allowed = intersectIntervals(allowed,
                                     {{-std::numeric_limits<double>::infinity(), innerRoots->min},
                                      {innerRoots->max, std::numeric_limits<double>::infinity()}});
      }
      return allowed;
    }

    double sampleTheta(CLHEP::HepRandomEngine* engine, const std::vector<Interval>& slopes) {
      double totalThetaLength = 0.;
      for (const auto& slope : slopes) {
        totalThetaLength += std::atan(slope.max) - std::atan(slope.min);
      }

      double offset = CLHEP::RandFlat::shoot(engine, 0., totalThetaLength);
      for (const auto& slope : slopes) {
        const double thetaMin = std::atan(slope.min);
        const double length = std::atan(slope.max) - thetaMin;
        if (offset <= length) {
          return thetaMin + offset;
        }
        offset -= length;
      }
      return std::atan(slopes.back().max);
    }

    double sampleRadius(CLHEP::HepRandomEngine* engine, const PlaneParameters& plane, RadialDistribution distribution) {
      if (distribution == RadialDistribution::kUniformArea) {
        return std::sqrt(CLHEP::RandFlat::shoot(engine, plane.rMin * plane.rMin, plane.rMax * plane.rMax));
      }
      return CLHEP::RandFlat::shoot(engine, plane.rMin, plane.rMax);
    }

    Point projectToZ(const Point& sampled, double sampledZ, double z, double theta, double momentumPhi) {
      const double transverseDisplacement = (z - sampledZ) * std::tan(theta);
      return {sampled.x + transverseDisplacement * std::cos(momentumPhi),
              sampled.y + transverseDisplacement * std::sin(momentumPhi)};
    }

    bool phiIsWithin(double phi, double min, double max) {
      constexpr double kTolerance = 1e-12;
      constexpr double kTwoPi = 2. * std::numbers::pi;
      if (max - min >= kTwoPi - kTolerance) {
        return true;
      }
      const double equivalentPhi = phi + kTwoPi * std::ceil((min - phi) / kTwoPi);
      return equivalentPhi <= max + kTolerance;
    }

    bool capContains(const Point& point, const PlaneParameters& cap) {
      constexpr double kTolerance = 1e-10;
      const double radius = std::hypot(point.x, point.y);
      return radius >= cap.rMin - kTolerance && radius <= cap.rMax + kTolerance &&
             phiIsWithin(std::atan2(point.y, point.x), cap.phiMin, cap.phiMax);
    }

    void validateParticleCompatibility(const ParticleGunParameters& parameters, double mass, double charge) {
      if (parameters.geometry.target && charge != 0.) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Target constraints assume a straight trajectory and therefore require a neutral particle.";
      }
      const auto& magnitude = parameters.momentum.magnitude;
      if (magnitude.variable == MagnitudeVariable::kEnergy && magnitude.min <= mass) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Momentum.Magnitude.Min must be greater than the particle mass when Variable is 'energy'. Min="
            << magnitude.min << " GeV, mass=" << mass << " GeV.";
      }
    }

    ResolvedParticle resolveParticle(CLHEP::HepRandomEngine* engine,
                                     const ParticleGunParameters& parameters,
                                     double mass) {
      const auto& magnitude = parameters.momentum.magnitude;
      const auto& direction = parameters.momentum.direction;
      const auto& geometry = parameters.geometry;
      const auto& origin = geometry.origin;
      const auto& production = geometry.production;
      const auto& sampledCap = geometry.sampleAt == SampleAt::kOrigin ? origin : production;

      const double sampledR = sampleRadius(engine, sampledCap, geometry.radialDistribution);
      const double sampledSpatialPhi = CLHEP::RandFlat::shoot(engine, sampledCap.phiMin, sampledCap.phiMax);
      const Point sampledPoint{sampledR * std::cos(sampledSpatialPhi), sampledR * std::sin(sampledSpatialPhi)};

      double theta = 0.;
      double momentumPhi = 0.;
      Point originPoint{};
      Point productionPoint{};
      std::optional<Point> targetPoint;
      bool accepted = false;

      for (unsigned int itry = 0; itry < parameters.maxDirectionTries; ++itry) {
        momentumPhi = CLHEP::RandFlat::shoot(engine, direction.phiMin, direction.phiMax);
        std::vector<Interval> allowedSlopes{{std::tan(direction.thetaMin), std::tan(direction.thetaMax)}};
        const double deltaPhi = momentumPhi - sampledSpatialPhi;

        const auto addRadialConstraint = [&](const PlaneParameters& cap) {
          allowedSlopes =
              intersectIntervals(allowedSlopes, allowedSlopesForCap(sampledR, deltaPhi, cap.z - sampledCap.z, cap));
        };

        addRadialConstraint(geometry.sampleAt == SampleAt::kOrigin ? production : origin);
        if (geometry.target) {
          addRadialConstraint(*geometry.target);
        }
        if (allowedSlopes.empty()) {
          continue;
        }

        theta = sampleTheta(engine, allowedSlopes);
        originPoint = projectToZ(sampledPoint, sampledCap.z, origin.z, theta, momentumPhi);
        productionPoint = projectToZ(sampledPoint, sampledCap.z, production.z, theta, momentumPhi);
        if (geometry.target) {
          targetPoint = projectToZ(sampledPoint, sampledCap.z, geometry.target->z, theta, momentumPhi);
        }

        accepted = capContains(originPoint, origin) && capContains(productionPoint, production) &&
                   (!geometry.target || capContains(*targetPoint, *geometry.target));
        if (accepted) {
          break;
        }
      }

      if (!accepted) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Failed to find a direction satisfying all configured caps after MaxDirectionTries="
            << parameters.maxDirectionTries
            << ". Fixed sampled point: cap=" << (geometry.sampleAt == SampleAt::kOrigin ? "Origin" : "Production")
            << ", R=" << sampledR << " cm, phi=" << sampledSpatialPhi << ".";
      }

      const double sampledMagnitude = CLHEP::RandFlat::shoot(engine, magnitude.min, magnitude.max);
      double pt = 0.;
      double pz = 0.;
      double energy = 0.;
      if (magnitude.variable == MagnitudeVariable::kPt) {
        pt = sampledMagnitude;
        pz = pt / std::tan(theta);
        energy = std::sqrt(pt * pt + pz * pz + mass * mass);
      } else {
        energy = sampledMagnitude;
        const double momentum = std::sqrt(energy * energy - mass * mass);
        pt = momentum * std::sin(theta);
        pz = momentum * std::cos(theta);
      }

      const FourMomentum momentum{pt * std::cos(momentumPhi), pt * std::sin(momentumPhi), pz, energy};

      double time = 0.;
      if (geometry.sampleAt == SampleAt::kOrigin) {
        const double deltaX = productionPoint.x - originPoint.x;
        const double deltaY = productionPoint.y - originPoint.y;
        const double deltaZ = production.z - origin.z;
        const double pathLength = std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
        const double absoluteMomentum =
            std::sqrt(momentum.px * momentum.px + momentum.py * momentum.py + momentum.pz * momentum.pz);
        time = pathLength * CLHEP::cm * energy / absoluteMomentum;
      }

      return {momentum, {productionPoint.x, productionPoint.y, production.z, time}};
    }

  }  // namespace

  class DisplacedParticleGunProducer : public edm::global::EDProducer<> {
  public:
    explicit DisplacedParticleGunProducer(const ParameterSet&);
    ~DisplacedParticleGunProducer() override = default;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const override;

    const ProducerParameters fParameters;
    const ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> fPDGTableToken;
  };

  DisplacedParticleGunProducer::DisplacedParticleGunProducer(const ParameterSet& pset)
      : fParameters(pset), fPDGTableToken(esConsumes<>()) {
    Service<RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
          << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
             "which appears to be absent.  Please add that service to your configuration\n"
             "or remove the modules that require it.";
    }

    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();
  }

  void DisplacedParticleGunProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::ParameterSetDescription pgun;

    edm::ParameterSetDescription magnitude;
    magnitude.add<std::string>("Variable");
    magnitude.add<double>("Min");
    magnitude.add<double>("Max");

    edm::ParameterSetDescription direction;
    direction.add<double>("ThetaMin");
    direction.add<double>("ThetaMax");
    direction.add<double>("PhiMin");
    direction.add<double>("PhiMax");

    edm::ParameterSetDescription momentum;
    momentum.add<edm::ParameterSetDescription>("Magnitude", magnitude);
    momentum.add<edm::ParameterSetDescription>("Direction", direction);

    edm::ParameterSetDescription origin;
    origin.add<double>("RMin");
    origin.add<double>("RMax");
    origin.add<double>("PhiMin");
    origin.add<double>("PhiMax");

    edm::ParameterSetDescription production;
    production.add<double>("Z");
    production.add<double>("RMin");
    production.add<double>("RMax");
    production.add<double>("PhiMin");
    production.add<double>("PhiMax");

    edm::ParameterSetDescription target;
    target.add<double>("Z");
    target.add<double>("RMin");
    target.add<double>("RMax");
    target.add<double>("PhiMin");
    target.add<double>("PhiMax");

    edm::ParameterSetDescription geometry;
    geometry.add<std::string>("SampleAt");
    geometry.add<std::string>("RadialDistribution");
    geometry.add<edm::ParameterSetDescription>("Origin", origin);
    geometry.add<edm::ParameterSetDescription>("Production", production);
    geometry.addOptional<edm::ParameterSetDescription>("Target", target);

    pgun.add<int>("PartID");
    pgun.add<int>("NParticles");
    pgun.add<edm::ParameterSetDescription>("Momentum", momentum);
    pgun.add<edm::ParameterSetDescription>("Geometry", geometry);
    pgun.add<unsigned int>("MaxDirectionTries");

    desc.add<edm::ParameterSetDescription>("PGunParameters", pgun);

    desc.addUntracked<int>("Verbosity", 0);

    descriptions.add("DisplacedParticleGunProducer", desc);
  }

  void DisplacedParticleGunProducer::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const {
    const auto& particleGun = fParameters.particleGun;

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    if (fParameters.verbosity > 0) {
      LogDebug("DisplacedParticleGunProducer")
          << " DisplacedParticleGunProducer : Begin New Event Generation" << std::endl;
    }

    HepMC::GenEvent* fEvt = new HepMC::GenEvent();

    auto const& pdgTable = es.getData(fPDGTableToken);
    const HepPDT::ParticleData* pData = pdgTable.particle(HepPDT::ParticleID(std::abs(particleGun.partId)));
    if (!pData) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Particle ID " << particleGun.partId << " not found in PDG table";
    }
    const double mass = pData->mass().value();
    validateParticleCompatibility(particleGun, mass, pData->charge());

    int barcode = 1;

    for (int ip = 0; ip < particleGun.nParticles; ++ip) {
      const ResolvedParticle resolved = resolveParticle(engine, particleGun, mass);
      const auto& momentum = resolved.momentum;
      const auto& vertex = resolved.vertex;

      auto* vtx = new HepMC::GenVertex(
          HepMC::FourVector(vertex.x * CLHEP::cm, vertex.y * CLHEP::cm, vertex.z * CLHEP::cm, vertex.time));
      auto* part = new HepMC::GenParticle(
          HepMC::FourVector(momentum.px, momentum.py, momentum.pz, momentum.energy), particleGun.partId, 1);
      part->suggest_barcode(barcode++);

      vtx->add_particle_out(part);
      fEvt->add_vertex(vtx);

      if (fParameters.verbosity > 0) {
        vtx->print();
        part->print();
      }
    }

    fEvt->set_event_number(e.id().event());
    fEvt->set_signal_process_id(20);

    if (fParameters.verbosity > 0) {
      fEvt->print();
    }

    auto bProduct = std::make_unique<HepMCProduct>();
    bProduct->addHepMCData(fEvt);
    e.put(std::move(bProduct), "unsmeared");

    auto genEventInfo = std::make_unique<GenEventInfoProduct>(fEvt);
    e.put(std::move(genEventInfo));

    if (fParameters.verbosity > 0) {
      std::cout << " DisplacedParticleGunProducer : Event Generation Done. " << std::endl;
    }
  }

}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::DisplacedParticleGunProducer;
DEFINE_FWK_MODULE(DisplacedParticleGunProducer);
