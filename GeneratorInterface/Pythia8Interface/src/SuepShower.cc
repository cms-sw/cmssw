// This is file contains the code to generate a dark sector shower in a strongly coupled, quasi-conformal hidden valley, often
// referred to as "soft unclustered energy patterns (SUEP)" or "softbomb" events. The shower is generated in its rest frame and
// for a realistic simulation this class needs to be interfaced with an event generator such as madgraph, pythia or herwig.
//
// The algorithm relies on arXiv:1305.5226. See arXiv:1612.00850 for a description of the model.
// Please cite both papers when using this code.
//
//
// Applicability:
// The ratio of the parameters m and T (m/T) should be an O(1) number. For m/T>>1 and m/T<<1 the theoretical description of the shower is not valid.
// The mass of the scalar which initiates the shower should be much larger than the mass of the mesons, in other words M>>m,T, by at least an order of magnitude.
//
// Written by Simon Knapen on 12/22/2019, following 1205.5226
// Adapted by Carlos Erice for CMSSW
#include "GeneratorInterface/Pythia8Interface/interface/SuepShower.h"

// constructor
SuepShower::SuepShower(double mass, double temperature, Pythia8::Rndm* rndmPtr) {
  edm::LogInfo("Pythia8Interface") << "Creating a SuepShower module";
  // Parameter setup
  darkmeson_mass_ = mass;
  fRndmPtr_ = rndmPtr;
  // No need to save temperature, as everything depends on it through m/T
  mass_over_T_ = darkmeson_mass_ / temperature;

  // 128 bit numerical precision, i.e. take machine precision
  tolerance_ = boost::math::tools::eps_tolerance<double>(128);

  // Median momentum
  p_m_ = sqrt(2 / (mass_over_T_ * mass_over_T_) * (1 + sqrt(1 + mass_over_T_ * mass_over_T_)));
  double pmax = sqrt(2 + 2 * sqrt(1 + mass_over_T_ * mass_over_T_)) / mass_over_T_;
  // Find the two cases where f(p)/f(pmax) = 1/e, given MB distribution, one appears at each side of pmax always
  p_plus_ = (boost::math::tools::bisect(
                 boost::bind(&SuepShower::logTestFunction, this, boost::placeholders::_1), pmax, 50.0, tolerance_))
                .first;  // first root
  p_minus_ = (boost::math::tools::bisect(
                  boost::bind(&SuepShower::logTestFunction, this, boost::placeholders::_1), 0.0, pmax, tolerance_))
                 .first;  // second root
  // Define the auxiliar quantities for the random generation of momenta
  lambda_plus_ = -fMaxwellBoltzmann(p_plus_) / fMaxwellBoltzmannPrime(p_plus_);
  lambda_minus_ = fMaxwellBoltzmann(p_minus_) / fMaxwellBoltzmannPrime(p_minus_);
  q_plus_ = lambda_plus_ / (p_plus_ - p_minus_);
  q_minus_ = lambda_minus_ / (p_plus_ - p_minus_);
  q_m_ = 1 - (q_plus_ + q_minus_);
}

SuepShower::~SuepShower() {}

// Generate a shower in the rest frame of the mediator
std::vector<Pythia8::Vec4> SuepShower::generateShower(double energy) {
  mediator_energy_ = energy;
  std::vector<Pythia8::Vec4> shower;
  double shower_energy = 0.0;

  // Fill up shower record
  while (shower_energy < mediator_energy_ || shower.size() < 2) {
    shower.push_back(generateFourVector());
    shower_energy += (shower.back()).e();
  }

  // Reballance momenta to ensure conservation
  // Correction starts at 0
  Pythia8::Vec4 correction = Pythia8::Vec4(0., 0., 0., 0.);
  for (const auto& daughter : shower) {
    correction = correction + daughter;
  }
  correction = correction / shower.size();
  // We only want to correct momenta first
  correction.e(0);
  for (auto& daughter : shower) {
    daughter = daughter - correction;
  }

  // With momentum conserved, balance energy. scale is the multiplicative factor needed such that sum_daughters((scale*p)^2+m^2) = E_parent, i.e. energy is conserved
  double scale;
  double minscale = 0.0;
  double maxscale = 2.0;
  while (SuepShower::reballanceFunction(minscale, shower) * SuepShower::reballanceFunction(maxscale, shower) > 0) {
    minscale = maxscale;
    maxscale *= 2;
  }

  scale =
      (boost::math::tools::bisect(boost::bind(&SuepShower::reballanceFunction, this, boost::placeholders::_1, shower),
                                  minscale,
                                  maxscale,
                                  tolerance_))
          .first;

  for (auto& daughter : shower) {
    daughter.px(daughter.px() * scale);
    daughter.py(daughter.py() * scale);
    daughter.pz(daughter.pz() * scale);
    // Force everything to be on-shell
    daughter.e(sqrt(daughter.pAbs2() + darkmeson_mass_ * darkmeson_mass_));
  }
  return shower;
}

//////// Private Methods ////////
// Maxwell-boltzman distribution, slightly massaged
const double SuepShower::fMaxwellBoltzmann(double p) {
  return p * p * exp(-mass_over_T_ * p * p / (1 + sqrt(1 + p * p)));
}

// Derivative of maxwell-boltzmann
const double SuepShower::fMaxwellBoltzmannPrime(double p) {
  return exp(-mass_over_T_ * p * p / (1 + sqrt(1 + p * p))) * p * (2 - mass_over_T_ * p * p / sqrt(1 + p * p));
}

// Test function to be solved for p_plus_ and p_minus_
const double SuepShower::logTestFunction(double p) { return log(fMaxwellBoltzmann(p) / fMaxwellBoltzmann(p_m_)) + 1.0; }

// Generate one random 4 std::vector from the thermal distribution
const Pythia8::Vec4 SuepShower::generateFourVector() {
  double en, phi, theta, momentum;  //kinematic variables of the 4 std::vector

  // First do momentum, following naming of arxiv:1305.5226
  double U, V, X, Y, E;
  int i = 0;
  while (i < 100) {
    // Very rarely (<1e-5 events) takes more than one loop to converge, set to 100 for safety
    U = fRndmPtr_->flat();
    V = fRndmPtr_->flat();

    if (U < q_m_) {
      Y = U / q_m_;
      X = (1 - Y) * (p_minus_ + lambda_minus_) + Y * (p_plus_ - lambda_plus_);
      if (V < fMaxwellBoltzmann(X) / fMaxwellBoltzmann(p_m_) && X > 0) {
        break;
      }
    } else {
      if (U < q_m_ + q_plus_) {
        E = -log((U - q_m_) / q_plus_);
        X = p_plus_ - lambda_plus_ * (1 - E);
        if (V < exp(E) * fMaxwellBoltzmann(X) / fMaxwellBoltzmann(p_m_) && X > 0) {
          break;
        }
      } else {
        E = -log((U - (q_m_ + q_plus_)) / q_minus_);
        X = p_minus_ + lambda_minus_ * (1 - E);
        if (V < exp(E) * fMaxwellBoltzmann(X) / fMaxwellBoltzmann(p_m_) && X > 0) {
          break;
        }
      }
    }
  }
  // X is the dimensionless momentum, p/m
  momentum = X * darkmeson_mass_;

  // now do the angles, isotropic
  phi = 2.0 * M_PI * (fRndmPtr_->flat());
  theta = acos(2.0 * (fRndmPtr_->flat()) - 1.0);

  // compose the 4 std::vector
  en = sqrt(momentum * momentum + darkmeson_mass_ * darkmeson_mass_);
  Pythia8::Vec4 daughterFourMomentum =
      Pythia8::Vec4(momentum * cos(phi) * sin(theta), momentum * sin(phi) * sin(theta), momentum * sin(theta), en);
  return daughterFourMomentum;
}

// Auxiliary function, to be solved in order to impose energy conservation
// To ballance energy, we solve for "scale" by demanding that this function vanishes, i.e. that shower energy and mediator energy are equal
// By rescaling the momentum rather than the energy, avoid having to do these annoying rotations from the previous version
const double SuepShower::reballanceFunction(double scale, const std::vector<Pythia8::Vec4>& shower) {
  double showerEnergy = 0.0;
  for (const auto& daughter : shower) {
    showerEnergy += sqrt(scale * scale * daughter.pAbs2() + darkmeson_mass_ * darkmeson_mass_);
  }
  return showerEnergy - mediator_energy_;
}
