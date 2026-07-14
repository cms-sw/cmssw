#include <algorithm>
#include <cmath>
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
        if (thetaMin < 0. || thetaMax >= std::numbers::pi / 2.) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Momentum.Direction theta bounds must lie inside [0, pi/2).";
        }
        if (phiMin < -std::numbers::pi || phiMax > std::numbers::pi) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Momentum.Direction phi bounds must lie inside [-pi, pi].";
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
        if (phiMin < -std::numbers::pi || phiMax > std::numbers::pi) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Geometry." << name << " phi bounds must lie inside [-pi, pi].";
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
          : radialDistribution(readRadialDistribution(pset.getParameter<std::string>("RadialDistribution"))),
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

      RadialDistribution radialDistribution;
      PlaneParameters origin;
      PlaneParameters production;
      std::optional<PlaneParameters> target;
    };

    double getDistanceBetweenIntervals(double firstMin, double firstMax, double secondMin, double secondMax) {
      if (firstMax < secondMin) {
        return secondMin - firstMax;
      }
      if (secondMax < firstMin) {
        return firstMin - secondMax;
      }
      return 0.;
    }

    bool isRadialRangeReachableForDirection(const PlaneParameters& initial,
                                            const PlaneParameters& final,
                                            const DirectionParameters& direction) {
      const double deltaZ = std::abs(final.z - initial.z);
      const double displacementMin = deltaZ * std::tan(direction.thetaMin);
      const double displacementMax = deltaZ * std::tan(direction.thetaMax);
      const double reachableRMin =
          getDistanceBetweenIntervals(initial.rMin, initial.rMax, displacementMin, displacementMax);
      const double reachableRMax = initial.rMax + displacementMax;
      return final.rMax >= reachableRMin && final.rMin <= reachableRMax;
    }

    void validateRadialReachability(const PlaneParameters& initial,
                                    const PlaneParameters& final,
                                    const DirectionParameters& direction,
                                    const char* initialName,
                                    const char* finalName) {
      if (!isRadialRangeReachableForDirection(initial, final, direction)) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Geometry." << finalName << " radial range is unreachable from Geometry." << initialName
            << " within the theta range.";
      }
    }

    struct ParticleGunParameters {
      explicit ParticleGunParameters(const ParameterSet& pset)
          : partId(pset.getParameter<int>("PartID")),
            nParticles(pset.getParameter<int>("NParticles")),
            momentum(pset.getParameter<ParameterSet>("Momentum")),
            geometry(pset.getParameter<ParameterSet>("Geometry")),
            maxSamplingAttempts(pset.getParameter<unsigned int>("MaxSamplingAttempts")) {
        if (nParticles <= 0) {
          throw cms::Exception("DisplacedParticleGunProducer") << "NParticles must be greater than zero.";
        }
        if (maxSamplingAttempts == 0) {
          throw cms::Exception("DisplacedParticleGunProducer") << "MaxSamplingAttempts must be greater than zero.";
        }
        if (momentum.magnitude.variable == MagnitudeVariable::kPt && momentum.direction.thetaMin == 0.) {
          throw cms::Exception("DisplacedParticleGunProducer")
              << "Momentum.Direction.ThetaMin must be greater than zero when Momentum.Magnitude.Variable is 'pt'.";
        }

        validateRadialReachability(geometry.origin, geometry.production, momentum.direction, "Origin", "Production");
        if (geometry.target) {
          validateRadialReachability(geometry.origin, *geometry.target, momentum.direction, "Origin", "Target");
          validateRadialReachability(geometry.production, *geometry.target, momentum.direction, "Production", "Target");
        }
      }

      int partId;
      int nParticles;
      MomentumParameters momentum;
      GeometryParameters geometry;
      unsigned int maxSamplingAttempts;
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
      double z;
    };

    struct SampledDirection {
      double theta;
      double phi;
      Point productionPoint;
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
      int pdgId;
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

    std::vector<Interval> subtractInterval(const Interval& allowed, const Interval& excluded) {
      std::vector<Interval> result;

      const double leftMax = std::min(allowed.max, excluded.min);
      if (allowed.min < leftMax) {
        result.push_back({allowed.min, leftMax});
      }

      const double rightMin = std::max(allowed.min, excluded.max);
      if (rightMin < allowed.max) {
        result.push_back({rightMin, allowed.max});
      }

      return result;
    }

    std::optional<Interval> getQuadraticRoots(double a, double b, double c) {
      const double discriminant = b * b - 4. * a * c;
      if (discriminant <= 0.) {
        return std::nullopt;
      }

      const double sqrtDiscriminant = std::sqrt(discriminant);
      return Interval{(-b - sqrtDiscriminant) / (2. * a), (-b + sqrtDiscriminant) / (2. * a)};
    }

    std::vector<Interval> getAllowedSlopesForCap(const Point& originPoint,
                                              double momentumPhi,
                                              const PlaneParameters& cap) {
      const double deltaZ = cap.z - originPoint.z;
      const double a = deltaZ * deltaZ;
      const double b = 2. * deltaZ * (originPoint.x * std::cos(momentumPhi) + originPoint.y * std::sin(momentumPhi));
      const double c = originPoint.x * originPoint.x + originPoint.y * originPoint.y;

      const auto outerRoots = getQuadraticRoots(a, b, c - cap.rMax * cap.rMax);
      if (!outerRoots) {
        return {};
      }

      const auto innerRoots = getQuadraticRoots(a, b, c - cap.rMin * cap.rMin);
      if (!innerRoots) {
        return {*outerRoots};
      }

      return subtractInterval(*outerRoots, *innerRoots);
    }

    std::vector<Interval> constrainSlopesToCap(const std::vector<Interval>& slopes,
                                               const Point& originPoint,
                                               double momentumPhi,
                                               const PlaneParameters& cap) {
      return intersectIntervals(slopes, getAllowedSlopesForCap(originPoint, momentumPhi, cap));
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

    Point projectToZ(const Point& sampled, double z, double momentumTheta, double momentumPhi) {
      const double transverseDisplacement = (z - sampled.z) * std::tan(momentumTheta);
      return {sampled.x + transverseDisplacement * std::cos(momentumPhi),
              sampled.y + transverseDisplacement * std::sin(momentumPhi),
              z};
    }

    bool isPhiWithin(double phi, double min, double max) { return phi >= min && phi <= max; }

    bool doesCapContain(const Point& point, const PlaneParameters& cap) {
      const double radius = std::hypot(point.x, point.y);
      return radius >= cap.rMin && radius <= cap.rMax &&
             isPhiWithin(std::atan2(point.y, point.x), cap.phiMin, cap.phiMax);
    }

    double resolveTimeOfFlight(const Point& origin, const Point& destination, const FourMomentum& momentum) {
      const double deltaX = destination.x - origin.x;
      const double deltaY = destination.y - origin.y;
      const double deltaZ = destination.z - origin.z;
      const double pathLength = std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
      const double absoluteMomentum =
          std::sqrt(momentum.px * momentum.px + momentum.py * momentum.py + momentum.pz * momentum.pz);
      return pathLength * CLHEP::cm * momentum.energy / absoluteMomentum;
    }

    std::optional<SampledDirection> sampleDirection(CLHEP::HepRandomEngine* engine,
                                                    const ParticleGunParameters& parameters,
                                                    const Point& originPoint) {
      const auto& direction = parameters.momentum.direction;
      const auto& geometry = parameters.geometry;

      for (unsigned int attempt = 0; attempt < parameters.maxSamplingAttempts; ++attempt) {
        const double phi = CLHEP::RandFlat::shoot(engine, direction.phiMin, direction.phiMax);
        std::vector<Interval> allowedSlopes{{std::tan(direction.thetaMin), std::tan(direction.thetaMax)}};

        allowedSlopes = constrainSlopesToCap(allowedSlopes, originPoint, phi, geometry.production);
        if (geometry.target) {
          allowedSlopes = constrainSlopesToCap(allowedSlopes, originPoint, phi, *geometry.target);
        }
        if (allowedSlopes.empty()) {
          continue;
        }

        const double theta = sampleTheta(engine, allowedSlopes);
        const Point productionPoint = projectToZ(originPoint, geometry.production.z, theta, phi);
        if (!doesCapContain(productionPoint, geometry.production)) {
          continue;
        }

        if (geometry.target) {
          const Point targetPoint = projectToZ(originPoint, geometry.target->z, theta, phi);
          if (!doesCapContain(targetPoint, *geometry.target)) {
            continue;
          }
        }

        return SampledDirection{theta, phi, productionPoint};
      }

      return std::nullopt;
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
      const auto& geometry = parameters.geometry;
      const auto& origin = geometry.origin;

      const double sampledR = sampleRadius(engine, origin, geometry.radialDistribution);
      const double sampledSpatialPhi = CLHEP::RandFlat::shoot(engine, origin.phiMin, origin.phiMax);
      const Point sampledOriginPoint{
          sampledR * std::cos(sampledSpatialPhi), sampledR * std::sin(sampledSpatialPhi), origin.z};

      const auto sampledDirection = sampleDirection(engine, parameters, sampledOriginPoint);
      if (!sampledDirection) {
        throw cms::Exception("DisplacedParticleGunProducer")
            << "Failed to find a direction satisfying all configured caps after MaxSamplingAttempts="
            << parameters.maxSamplingAttempts << ". Fixed sampled point: cap=Origin, R=" << sampledR
            << " cm, phi=" << sampledSpatialPhi << ".";
      }
      const auto [theta, momentumPhi, productionPoint] = *sampledDirection;

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

      const double time = resolveTimeOfFlight(sampledOriginPoint, productionPoint, momentum);

      return {parameters.partId, momentum, {productionPoint.x, productionPoint.y, productionPoint.z, time}};
    }

    void appendParticleToGenEvent(HepMC::GenEvent& genEvent,
                                  const ResolvedParticle& particle,
                                  int barcode,
                                  bool verbose) {
      const auto& momentum = particle.momentum;
      const auto& vertex = particle.vertex;

      auto* genVertex = new HepMC::GenVertex(
          HepMC::FourVector(vertex.x * CLHEP::cm, vertex.y * CLHEP::cm, vertex.z * CLHEP::cm, vertex.time));
      auto* genParticle = new HepMC::GenParticle(
          HepMC::FourVector(momentum.px, momentum.py, momentum.pz, momentum.energy), particle.pdgId, 1);
      genParticle->suggest_barcode(barcode);

      genVertex->add_particle_out(genParticle);
      genEvent.add_vertex(genVertex);

      if (verbose) {
        genVertex->print();
        genParticle->print();
      }
    }

  }  // namespace

  class DisplacedParticleGunProducer : public edm::global::EDProducer<> {
  public:
    explicit DisplacedParticleGunProducer(const ParameterSet&);
    ~DisplacedParticleGunProducer() override = default;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const override;

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
    geometry.add<std::string>("RadialDistribution");
    geometry.add<edm::ParameterSetDescription>("Origin", origin);
    geometry.add<edm::ParameterSetDescription>("Production", production);
    geometry.addOptional<edm::ParameterSetDescription>("Target", target);

    pgun.add<int>("PartID");
    pgun.add<int>("NParticles");
    pgun.add<edm::ParameterSetDescription>("Momentum", momentum);
    pgun.add<edm::ParameterSetDescription>("Geometry", geometry);
    pgun.add<unsigned int>("MaxSamplingAttempts");

    desc.add<edm::ParameterSetDescription>("PGunParameters", pgun);

    desc.addUntracked<int>("Verbosity", 0);

    descriptions.add("DisplacedParticleGunProducer", desc);
  }

  void DisplacedParticleGunProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
    if (fParameters.verbosity > 0) {
      LogDebug("DisplacedParticleGunProducer")
          << " DisplacedParticleGunProducer : Begin New Event Generation" << std::endl;
    }

    const auto& particleGun = fParameters.particleGun;
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* randomEngine = &rng->getEngine(event.streamID());

    auto const& pdgTable = setup.getData(fPDGTableToken);
    const HepPDT::ParticleData* pData = pdgTable.particle(HepPDT::ParticleID(std::abs(particleGun.partId)));
    if (!pData) {
      throw cms::Exception("DisplacedParticleGunProducer")
          << "Particle ID " << particleGun.partId << " not found in PDG table";
    }

    const double mass = pData->mass().value();
    validateParticleCompatibility(particleGun, mass, pData->charge());

    HepMC::GenEvent* genEvent = new HepMC::GenEvent();
    genEvent->set_event_number(event.id().event());
    genEvent->set_signal_process_id(20);

    for (int particleIndex = 0; particleIndex < particleGun.nParticles; ++particleIndex) {
      const ResolvedParticle particle = resolveParticle(randomEngine, particleGun, mass);
      appendParticleToGenEvent(*genEvent, particle, particleIndex + 1, fParameters.verbosity > 0);
    }

    if (fParameters.verbosity > 0) {
      genEvent->print();
    }

    auto hepMcProduct = std::make_unique<HepMCProduct>();
    hepMcProduct->addHepMCData(genEvent);
    event.put(std::move(hepMcProduct), "unsmeared");

    auto genEventInfo = std::make_unique<GenEventInfoProduct>(genEvent);
    event.put(std::move(genEventInfo));

    if (fParameters.verbosity > 0) {
      LogDebug("DisplacedParticleGunProducer")
          << " DisplacedParticleGunProducer : Event Generation Done. " << std::endl;
    }
  }

}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::DisplacedParticleGunProducer;
DEFINE_FWK_MODULE(DisplacedParticleGunProducer);
