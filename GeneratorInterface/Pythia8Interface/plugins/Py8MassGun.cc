
#include <memory>

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8MassGun : public Py8GunBase {
  public:
    Py8MassGun(edm::ParameterSet const&);
    ~Py8MassGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // PtGun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinP;
    double fMaxP;
    double fMinPt;
    double fMaxPt;
    double fMinM;
    double fMaxM;
    int fMomMode;
  };

  // implementation
  //
  Py8MassGun::Py8MassGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinP = pgun_params.getParameter<double>("MinP");                                      // ,  0.);
    fMaxP = pgun_params.getParameter<double>("MaxP");                                      // ,  0.);
    fMinPt = pgun_params.getParameter<double>("MinPt");                                    // ,  0.);
    fMaxPt = pgun_params.getParameter<double>("MaxPt");                                    // ,  0.);
    fMinM = pgun_params.getParameter<double>("MinM");                                      // ,  0.);
    fMaxM = pgun_params.getParameter<double>("MaxM");                                      // ,  0.);
    fMomMode = pgun_params.getParameter<int>("MomMode");                                   // ,  1);
  }

  bool Py8MassGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();
    size_t pSize = fPartIDs.size();
    if (pSize > 2)
      return false;

    // Pick a flat mass range
    double phi, eta, the, ee, pp;
    double m0 = (fMaxM - fMinM) * randomEngine().flat() + fMinM;
    // Global eta
    eta = (fMaxEta - fMinEta) * randomEngine().flat() + fMinEta;

    if (pSize == 2) {
      // Masses.
      double m1 = fMasterGen->particleData.m0(fPartIDs[0]);
      double m2 = fMasterGen->particleData.m0(fPartIDs[1]);

      // Energies and absolute momentum in the rest frame.
      if (m1 + m2 > m0)
        return false;
      double e1 = 0.5 * (m0 * m0 + m1 * m1 - m2 * m2) / m0;
      double e2 = 0.5 * (m0 * m0 + m2 * m2 - m1 * m1) / m0;
      double pAbs = 0.5 * sqrt((m0 - m1 - m2) * (m0 + m1 + m2) * (m0 + m1 - m2) * (m0 - m1 + m2)) / m0;
      // Isotropic angles in rest frame give three-momentum.
      double cosTheta = 2. * randomEngine().flat() - 1.;
      double sinTheta = sqrt(1. - cosTheta * cosTheta);
      phi = 2. * M_PI * randomEngine().flat();

      double pX = pAbs * sinTheta * cos(phi);
      double pY = pAbs * sinTheta * sin(phi);
      double pZ = pAbs * cosTheta;

      (fMasterGen->event).append(fPartIDs[0], 1, 0, 0, pX, pY, pZ, e1, m1);
      (fMasterGen->event).append(fPartIDs[1], 1, 0, 0, -pX, -pY, -pZ, e2, m2);
    } else {
      (fMasterGen->event).append(fPartIDs[0], 1, 0, 0, 0.0, 0.0, 0.0, m0, m0);
    }

    //now the boost (from input params)
    if (fMomMode == 0) {
      pp = (fMaxP - fMinP) * randomEngine().flat() + fMinP;
    } else {
      double pT = (fMaxPt - fMinPt) * randomEngine().flat() + fMinPt;
      pp = pT * cosh(eta);
    }
    ee = sqrt(m0 * m0 + pp * pp);

    //the boost direction (from input params)
    //
    phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
    the = 2. * atan(exp(-eta));

    double betaX = pp / ee * std::sin(the) * std::cos(phi);
    double betaY = pp / ee * std::sin(the) * std::sin(phi);
    double betaZ = pp / ee * std::cos(the);

    // boost all particles
    //
    (fMasterGen->event).bst(betaX, betaY, betaZ);

    if (!fMasterGen->next())
      return false;

    event() = std::make_unique<HepMC::GenEvent>();
    return toHepMC.fill_next_event(fMasterGen->event, event().get());
  }

  const char* Py8MassGun::classname() const { return "Py8MassGun"; }

  typedef edm::GeneratorFilter<gen::Py8MassGun, gen::ExternalDecayDriver> Pythia8MassGun;

}  // namespace gen

using gen::Pythia8MassGun;
DEFINE_FWK_MODULE(Pythia8MassGun);
