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

#include <TMath.h>
#include <TF1.h>

///////////////////////////////////////////////
// Authors: Sandro Fonseca de Souza and Andre Sznajder (UERJ/Brazil)
// Date: 23-Nov-2010
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Sekmen, 18 May 2017
//
// Revision: Code very buggy, PetrukhinFunc return negative values, double bremProba wasn't properly defined etc.
//           Should be all fixed by now
//           S. Kurz, 23 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {
  //! Implementation of Bremsstrahlung from mu+/mu- in the tracker layers based on a Petrukhin Model (nuclear screening correction).
  /*!
        Computes the number, energy and angles of Bremsstrahlung photons emitted by muons
        and modifies mu+/mu- particle accordingly.
    */
  class MuonBremsstrahlung : public InteractionModel {
  public:
    //! Constructor.
    MuonBremsstrahlung(const std::string &name, const edm::ParameterSet &cfg);

    //! Default destructor.
    ~MuonBremsstrahlung() override { ; };

    //! Perform the interaction.
    /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
    void interact(Particle &particle,
                  const SimplifiedGeometry &layer,
                  std::vector<std::unique_ptr<Particle> > &secondaries,
                  const RandomEngineAndDistribution &random) override;

  private:
    //! Compute Brem photon energy and angles, if any.
    /*!
            \param particle The particle that interacts with the matter.
            \param xmin Minimum fraction of the particle's energy that has to be converted to a photon.
            \param random The Random Engine.
            \return Momentum 4-vector of a bremsstrahlung photon.
        */
    math::XYZTLorentzVector brem(Particle &particle, double xmin, const RandomEngineAndDistribution &random) const;

    //! A universal angular distribution.
    /*!
            \param ener 
            \param partm 
            \param efrac 
            \param random The Random Engine.
            \return Theta from universal distribution
        */
    double gbteth(const double ener,
                  const double partm,
                  const double efrac,
                  const RandomEngineAndDistribution &random) const;

    //! Petrukhin Function: Returns cross section using nuclear-electron screening correction from G4 style
    static double PetrukhinFunc(double *x, double *p);

    TF1 *Petrfunc;                    //!< The Petrukhin Function
    double minPhotonEnergy_;          //!< Cut on minimum energy of bremsstrahlung photons
    double minPhotonEnergyFraction_;  //!< Cut on minimum fraction of particle's energy which has to be carried by photon
    double density_;                  //!< Density of material (usually silicon rho=2.329)
    double radLenInCm_;               //!< Radiation length of material (usually silicon X0=9.360)
    double A_;                        //!< Atomic weight of material (usually silicon A=28.0855)
    double Z_;                        //!< Atomic number of material (usually silicon Z=14)
  };
}  // namespace fastsim

fastsim::MuonBremsstrahlung::MuonBremsstrahlung(const std::string &name, const edm::ParameterSet &cfg)
    : fastsim::InteractionModel(name) {
  // Set the minimal photon energy for a Brem from mu+/-
  minPhotonEnergy_ = cfg.getParameter<double>("minPhotonEnergy");
  minPhotonEnergyFraction_ = cfg.getParameter<double>("minPhotonEnergyFraction");
  // Material properties
  A_ = cfg.getParameter<double>("A");
  Z_ = cfg.getParameter<double>("Z");
  density_ = cfg.getParameter<double>("density");
  radLenInCm_ = cfg.getParameter<double>("radLen");
}

void fastsim::MuonBremsstrahlung::interact(fastsim::Particle &particle,
                                           const SimplifiedGeometry &layer,
                                           std::vector<std::unique_ptr<fastsim::Particle> > &secondaries,
                                           const RandomEngineAndDistribution &random) {
  // only consider muons
  if (std::abs(particle.pdgId()) != 13) {
    return;
  }

  double radLengths = layer.getThickness(particle.position(), particle.momentum());
  //
  // no material
  //
  if (radLengths < 1E-10) {
    return;
  }

  // Protection : Just stop the electron if more than 1 radiation lengths.
  // This case corresponds to an electron entering the layer parallel to
  // the layer axis - no reliable simulation can be done in that case...
  if (radLengths > 4.) {
    particle.momentum().SetXYZT(0., 0., 0., 0.);
    return;
  }

  // muon must have more energy than minimum photon energy
  if (particle.momentum().E() - particle.momentum().mass() < minPhotonEnergy_) {
    return;
  }

  // Min fraction of muon's energy transferred to the photon
  double xmin = std::max(minPhotonEnergy_ / particle.momentum().E(), minPhotonEnergyFraction_);
  // Hard brem probability with a photon Energy above threshold.
  if (xmin >= 1. || xmin <= 0.) {
    return;
  }

  // Max fraction of muon's energy transferred to the photon
  double xmax = 1.;

  // create TF1 using a free C function
  Petrfunc = new TF1("Petrfunc", PetrukhinFunc, xmin, xmax, 3);
  // Setting parameters
  Petrfunc->SetParameters(particle.momentum().E(), A_, Z_);
  // d = distance for several materials
  // X0 = radLen
  // d = radLengths * X0 (for tracker, yoke, ECAL and HCAL)
  double distance = radLengths * radLenInCm_;

  // Integration
  // Fixed previous version which used Petrfunc->Integral(0.,1.) -> does not make sense
  double bremProba = density_ * distance * (fastsim::Constants::NA / A_) * (Petrfunc->Integral(xmin, xmax));
  if (bremProba < 1E-10) {
    return;
  }

  // Number of photons to be radiated.
  unsigned int nPhotons = random.poissonShoot(bremProba);
  if (nPhotons == 0) {
    return;
  }

  //Rotate to the lab frame
  double theta = particle.momentum().Theta();
  double phi = particle.momentum().Phi();

  // Energy of these photons
  for (unsigned int i = 0; i < nPhotons; ++i) {
    // Throw momentum of the photon
    math::XYZTLorentzVector photonMom = brem(particle, xmin, random);

    // Check that there is enough energy left.
    if (particle.momentum().E() - particle.momentum().mass() < photonMom.E())
      break;

    // Rotate to the lab frame
    photonMom = ROOT::Math::RotationZ(phi) * (ROOT::Math::RotationY(theta) * photonMom);

    // Add a photon
    secondaries.emplace_back(new fastsim::Particle(22, particle.position(), photonMom));

    // Update the original e+/-
    particle.momentum() -= photonMom;
  }
}

math::XYZTLorentzVector fastsim::MuonBremsstrahlung::brem(fastsim::Particle &particle,
                                                          double xmin,
                                                          const RandomEngineAndDistribution &random) const {
  // This is a simple version of a Muon Brem using Petrukhin model.
  // Ref: http://pdg.lbl.gov/2008/AtomicNuclearProperties/adndt.pdf
  double xp = Petrfunc->GetRandom();

  // Have photon energy. Now generate angles with respect to the z axis
  // defined by the incoming particle's momentum.

  // Isotropic in phi
  const double phi = random.flatShoot() * 2. * M_PI;
  // theta from universal distribution
  const double theta = gbteth(particle.momentum().E(), fastsim::Constants::muMass, xp, random) *
                       fastsim::Constants::muMass / particle.momentum().E();

  // Make momentum components
  double stheta = std::sin(theta);
  double ctheta = std::cos(theta);
  double sphi = std::sin(phi);
  double cphi = std::cos(phi);

  return xp * particle.momentum().E() * math::XYZTLorentzVector(stheta * cphi, stheta * sphi, ctheta, 1.);
}

double fastsim::MuonBremsstrahlung::gbteth(const double ener,
                                           const double partm,
                                           const double efrac,
                                           const RandomEngineAndDistribution &random) const {
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

double fastsim::MuonBremsstrahlung::PetrukhinFunc(double *x, double *p) {
  // Function independent variable
  double nu = x[0];  // fraction of muon's energy transferred to the photon

  // Parameters
  double E = p[0];  // Muon Energy (in GeV)
  double A = p[1];  // Atomic weight
  double Z = p[2];  // Atomic number

  /*
        //////////////////////////////////////////////////
        // Function of Muom Brem using  nuclear screening correction
        // Ref: http://pdg.lbl.gov/2008/AtomicNuclearProperties/adndt.pdf
        // http://geant4.cern.ch/G4UsersDocuments/UsersGuides/PhysicsReferenceManual/html/node48.html

        // Physical constants
        double B = 182.7;
        double ee = sqrt(2.7181) ; // sqrt(e)
        double ZZ=  pow( Z,-1./3.); // Z^-1/3
        ///////////////////////////////////////////////////////////////////
        double emass = 0.0005109990615;  // electron mass (GeV/c^2)
        double mumass = 0.105658367;//mu mass  (GeV/c^2)

        double re = 2.817940285e-13;// Classical electron radius (Units: cm)
        double alpha = 1./137.03599976; // fine structure constant
        double Dn = 1.54* (pow(A,0.27));
        double constant =  pow((2.0 * Z * emass/mumass * re ),2.0);
        //////////////////////////////////////////////

        double delta = (mumass * mumass * nu) /(2.* E * (1.- nu)); 

        double Delta_n = TMath::Log(Dn / (1.+ delta *( Dn * ee -2.)/ mumass)); //nuclear screening correction 

        double Phi = TMath::Log((B * mumass * ZZ / emass)/ (1.+ delta * ee * B * ZZ  / emass)) - Delta_n;//phi(delta)

        // Diff. Cross Section for Muon Brem from a screened nuclear (Equation 16: REF: LBNL-44742)
        double f = alpha * constant *(4./3.-4./3.*nu + nu*nu)*Phi/nu;
    */

  //////////////////////////////////////////////////
  // Function for Muon Brem Xsec from G4
  //////////////////////////////////////////////////

  // Physical constants
  double B = 183.;
  double Bl = 1429.;
  double ee = 1.64872;            // sqrt(e)
  double Z13 = pow(Z, -1. / 3.);  // Z^-1/3
  double Z23 = pow(Z, -2. / 3.);  // Z^-2/3

  // Original values of paper
  double emass = 0.0005109990615;  // electron mass (GeV/c^2)
  double mumass = 0.105658367;     // muon mass  (GeV/c^2)
  // double re = 2.817940285e-13;     // Classical electron radius (Units: cm)
  double alpha = 0.00729735;      // 1./137.03599976; // fine structure constant
  double constant = 1.85736e-30;  // pow( ( emass / mumass * re ) , 2.0);

  // Use nomenclature from reference -> Follow those formula step by step
  if (nu * E >= E - mumass)
    return 0;

  double Dn = 1.54 * (pow(A, 0.27));
  double Dnl = pow(Dn, (1. - 1. / Z));

  double delta = (mumass * mumass * nu) / (2. * E * (1. - nu));

  double Phi_n = TMath::Log(B * Z13 * (mumass + delta * (Dnl * ee - 2)) / (Dnl * (emass + delta * ee * B * Z13)));
  if (Phi_n < 0)
    Phi_n = 0;

  double Phi_e =
      TMath::Log((Bl * Z23 * mumass) / (1. + delta * mumass / (emass * emass * ee)) / (emass + delta * ee * Bl * Z23));
  if (Phi_e < 0 || nu * E >= E / (1. + (mumass * mumass / (2. * emass * E))))
    Phi_e = 0;

  // Diff. Cross Section for Muon Brem from G4 (without NA/A factor)
  double f = 16. / 3. * alpha * constant * Z * (Z * Phi_n + Phi_e) * (1. / nu) * (1. - nu + 0.75 * nu * nu);

  return f;
}

DEFINE_EDM_PLUGIN(fastsim::InteractionModelFactory, fastsim::MuonBremsstrahlung, "fastsim::MuonBremsstrahlung");
