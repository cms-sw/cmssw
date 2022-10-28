#ifndef GeneratorInterface_Pythia8Interface_SuepShower_h
#define GeneratorInterface_Pythia8Interface_SuepShower_h

#include <vector>
#include <utility>
#include <cmath>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include "Pythia8/Pythia.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class SuepShower {
public:
  // Constructor
  SuepShower(double mass, double temperature, Pythia8::Rndm* rndmPtr);

  // Empty destructor
  ~SuepShower();

  // Method to generate 4-momenta of dark mesons after the showering
  std::vector<Pythia8::Vec4> generateShower(double energy);

private:
  // private variables
  // Shower parameters
  double darkmeson_mass_;
  double mass_over_T_;

  // Overall energy of the decaying particle
  double mediator_energy_;

  // For the numerical algorithm precision
  boost::math::tools::eps_tolerance<double> tolerance_;

  // Several auxiliar variables for generating the 4-momentum of showered particles. Following the naming of Appendix 1 of https://arxiv.org/pdf/1305.5226.pdf
  // Median momentum in the M-B distribution
  double p_m_;
  // Two values of momentum at fMB(x)/fMB(p_m_) = e
  double p_plus_, p_minus_;
  // Auxiliars: fMB(p_plus_)/f'(p_plus_),  fMB(p_minus_)/f'(p_minus_)
  double lambda_plus_, lambda_minus_;
  // More auxiliars: lambda_plus_/(p_plus_ + p_minus_), lambda_minus_/(p_plus_ + p_minus_), 1-q_plus_-q_minus_
  double q_plus_, q_minus_, q_m_;

  // Pythia random number generator, to get the randomness into the shower
  Pythia8::Rndm* fRndmPtr_;

  // Methods
  // Maxwell-Boltzmann distribution as a function of |p|, slightly massaged, Eq. 6 of https://arxiv.org/pdf/1305.5226.pdf
  const double fMaxwellBoltzmann(double p);
  // Maxwell-Boltzmann derivative as a function of |p|, slightly massaged
  const double fMaxwellBoltzmannPrime(double p);
  // Log(fMaxwellBoltzmann(x)/fMaxwellBoltzmann(xmedian))+1, as a function of |p|, to be solved to find p_plus_, p_minus_
  const double logTestFunction(double p);
  // Generate the four vector of a particle (dark meson) in the shower
  const Pythia8::Vec4 generateFourVector();
  // sum(scale*scale*p*p + m*m) - E*E, find roots in scale to reballance momenta and impose E conservation
  const double reballanceFunction(double scale, const std::vector<Pythia8::Vec4>& shower);
};

#endif
