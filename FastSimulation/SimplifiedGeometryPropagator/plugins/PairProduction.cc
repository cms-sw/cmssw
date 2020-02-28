#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <memory>

#include <Math/RotationY.h>
#include <Math/RotationZ.h>

#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "DataFormats/Math/interface/LorentzVector.h"

///////////////////////////////////////////////
// Author: Patrick Janot
// Date: 24-Dec-2003
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  //! Computes the probability for photons to convert into an e+e- pair in the tracker layer.
  /*!
        In case, it returns a list of two Secondaries (e+ and e-).
    */
  class PairProduction : public InteractionModel {
  public:
    //! Constructor.
    PairProduction(const std::string& name, const edm::ParameterSet& cfg);

    //! Default destructor.
    ~PairProduction() override { ; };

    //! Perform the interaction.
    /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
    void interact(fastsim::Particle& particle,
                  const SimplifiedGeometry& layer,
                  std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                  const RandomEngineAndDistribution& random) override;

  private:
    //! A universal angular distribution.
    /*!
            \param ener 
            \param partm 
            \param efrac 
            \param random The Random Engine.
            \return Theta from universal distribution
        */
    double gbteth(double ener, double partm, double efrac, const RandomEngineAndDistribution& random) const;

    double minPhotonEnergy_;  //!< Cut on minimum energy of photons
    double Z_;                //!< Atomic number of material (usually silicon Z=14)
  };
}  // namespace fastsim

fastsim::PairProduction::PairProduction(const std::string& name, const edm::ParameterSet& cfg)
    : fastsim::InteractionModel(name) {
  // Set the minimal photon energy for possible conversion
  minPhotonEnergy_ = cfg.getParameter<double>("photonEnergyCut");
  // Material properties
  Z_ = cfg.getParameter<double>("Z");
}

void fastsim::PairProduction::interact(fastsim::Particle& particle,
                                       const SimplifiedGeometry& layer,
                                       std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                                       const RandomEngineAndDistribution& random) {
  double eGamma = particle.momentum().e();
  //
  // only consider photons
  //
  if (particle.pdgId() != 22) {
    return;
  }

  double radLengths = layer.getThickness(particle.position(), particle.momentum());
  //
  // no material
  //
  if (radLengths < 1E-10) {
    return;
  }

  //
  // The photon has enough energy to create a pair
  //
  if (eGamma < minPhotonEnergy_) {
    return;
  }

  //
  // Probability to convert is 7/9*(dx/X0)
  //
  if (-std::log(random.flatShoot()) > (7. / 9.) * radLengths) {
    return;
  }

  double xe = 0;
  double eMass = fastsim::Constants::eMass;
  double xm = eMass / eGamma;
  double weight = 0.;

  // Generate electron energy between emass and eGamma-emass
  do {
    xe = random.flatShoot() * (1. - 2. * xm) + xm;
    weight = 1. - 4. / 3. * xe * (1. - xe);
  } while (weight < random.flatShoot());

  // the electron
  double eElectron = xe * eGamma;
  double tElectron = eElectron - eMass;
  double pElectron = std::sqrt(std::max((eElectron + eMass) * tElectron, 0.));

  // the positron
  double ePositron = eGamma - eElectron;
  double tPositron = ePositron - eMass;
  double pPositron = std::sqrt((ePositron + eMass) * tPositron);

  // Generate angles
  double phi = random.flatShoot() * 2. * M_PI;
  double sphi = std::sin(phi);
  double cphi = std::cos(phi);

  double stheta1, stheta2, ctheta1, ctheta2;

  if (eElectron > ePositron) {
    double theta1 = gbteth(eElectron, eMass, xe, random) * eMass / eElectron;
    stheta1 = std::sin(theta1);
    ctheta1 = std::cos(theta1);
    stheta2 = stheta1 * pElectron / pPositron;
    ctheta2 = std::sqrt(std::max(0., 1.0 - (stheta2 * stheta2)));
  } else {
    double theta2 = gbteth(ePositron, eMass, xe, random) * eMass / ePositron;
    stheta2 = std::sin(theta2);
    ctheta2 = std::cos(theta2);
    stheta1 = stheta2 * pPositron / pElectron;
    ctheta1 = std::sqrt(std::max(0., 1.0 - (stheta1 * stheta1)));
  }

  //Rotate to the lab frame
  double thetaLab = particle.momentum().Theta();
  double phiLab = particle.momentum().Phi();

  // Add a electron
  secondaries.emplace_back(new fastsim::Particle(
      11,
      particle.position(),
      math::XYZTLorentzVector(pElectron * stheta1 * cphi, pElectron * stheta1 * sphi, pElectron * ctheta1, eElectron)));
  secondaries.back()->momentum() =
      ROOT::Math::RotationZ(phiLab) * (ROOT::Math::RotationY(thetaLab) * secondaries.back()->momentum());

  // Add a positron
  secondaries.emplace_back(new fastsim::Particle(
      -11,
      particle.position(),
      math::XYZTLorentzVector(
          -pPositron * stheta2 * cphi, -pPositron * stheta2 * sphi, pPositron * ctheta2, ePositron)));
  secondaries.back()->momentum() =
      ROOT::Math::RotationZ(phiLab) * (ROOT::Math::RotationY(thetaLab) * secondaries.back()->momentum());

  // The photon converted
  particle.momentum().SetXYZT(0., 0., 0., 0.);
}

double fastsim::PairProduction::gbteth(const double ener,
                                       const double partm,
                                       const double efrac,
                                       const RandomEngineAndDistribution& random) const {
  // Details on implementation here
  // http://www.dnp.fmph.uniba.sk/cernlib/asdoc/geant_html3/node299.html#GBTETH
  // http://svn.cern.ch/guest/AliRoot/tags/v3-07-03/GEANT321/gphys/gbteth.F

  const double alfa = 0.625;

  const double d = 0.13 * (0.8 + 1.3 / Z_) * (100.0 + (1.0 / ener)) * (1.0 + efrac);
  const double w1 = 9.0 / (9.0 + d);
  const double umax = ener * M_PI / partm;
  double u;

  do {
    double beta = (random.flatShoot() <= w1) ? alfa : 3.0 * alfa;
    u = -std::log(random.flatShoot() * random.flatShoot()) / beta;
  } while (u >= umax);

  return u;
}

DEFINE_EDM_PLUGIN(fastsim::InteractionModelFactory, fastsim::PairProduction, "fastsim::PairProduction");
