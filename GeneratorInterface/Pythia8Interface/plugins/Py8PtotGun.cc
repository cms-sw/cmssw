#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8PtotGun : public Py8GunBase {
  public:
    Py8PtotGun(edm::ParameterSet const&);
    ~Py8PtotGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // Ptot Gun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinPtot;
    double fMaxPtot;
    bool fAddAntiParticle;
  };

  // implementation
  //
  Py8PtotGun::Py8PtotGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinPtot = pgun_params.getParameter<double>("MinPtot");                                // ,  0.);
    fMaxPtot = pgun_params.getParameter<double>("MaxPtot");                                // ,  0.);
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");                  //, false) ;
  }

  bool Py8PtotGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();

    for (size_t i = 0; i < fPartIDs.size(); i++) {
      int particleID = fPartIDs[i];  // this is PDG - need to convert to Py8 ???

      double phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
      double eta = (fMaxEta - fMinEta) * randomEngine().flat() + fMinEta;
      double the = 2. * atan(exp(-eta));

      double pp = (fMaxPtot - fMinPtot) * randomEngine().flat() + fMinPtot;

      double mass = (fMasterGen->particleData).m0(particleID);

      double pt = pp * sin(the);
      double ee = sqrt(pp * pp + mass * mass);
      double px = pt * cos(phi);
      double py = pt * sin(phi);
      double pz = pp * cos(the);

      if (!((fMasterGen->particleData).isParticle(particleID))) {
        particleID = std::abs(particleID);
      }
      if (1 <= std::abs(particleID) && std::abs(particleID) <= 6)  // quarks
        (fMasterGen->event).append(particleID, 23, 101, 0, px, py, pz, ee, mass);
      else if (std::abs(particleID) == 21)  // gluons
        (fMasterGen->event).append(21, 23, 101, 102, px, py, pz, ee, mass);
      // other
      else {
        (fMasterGen->event).append(particleID, 1, 0, 0, px, py, pz, ee, mass);
        int eventSize = (fMasterGen->event).size() - 1;
        // -log(flat) = exponential distribution
        double tauTmp = -(fMasterGen->event)[eventSize].tau0() * log(randomEngine().flat());
        (fMasterGen->event)[eventSize].tau(tauTmp);
      }

      // Here also need to add anti-particle (if any)
      // otherwise just add a 2nd particle of the same type
      // (for example, gamma)
      //
      if (fAddAntiParticle) {
        if (1 <= std::abs(particleID) && std::abs(particleID) <= 6) {  // quarks
          (fMasterGen->event).append(-particleID, 23, 0, 101, -px, -py, -pz, ee, mass);
        } else if (std::abs(particleID) == 21) {  // gluons
          (fMasterGen->event).append(21, 23, 102, 101, -px, -py, -pz, ee, mass);
        } else {
          if ((fMasterGen->particleData).isParticle(-particleID)) {
            (fMasterGen->event).append(-particleID, 1, 0, 0, -px, -py, -pz, ee, mass);
          } else {
            (fMasterGen->event).append(particleID, 1, 0, 0, -px, -py, -pz, ee, mass);
          }
          int eventSize = (fMasterGen->event).size() - 1;
          // -log(flat) = exponential distribution
          double tauTmp = -(fMasterGen->event)[eventSize].tau0() * log(randomEngine().flat());
          (fMasterGen->event)[eventSize].tau(tauTmp);
        }
      }
    }

    if (!fMasterGen->next())
      return false;
    evtGenDecay();

    event().reset(new HepMC::GenEvent);
    return toHepMC.fill_next_event(fMasterGen->event, event().get());
  }

  const char* Py8PtotGun::classname() const { return "Py8PtotGun"; }

  typedef edm::GeneratorFilter<gen::Py8PtotGun, gen::ExternalDecayDriver> Pythia8PtotGun;

}  // namespace gen

using gen::Pythia8PtotGun;
DEFINE_FWK_MODULE(Pythia8PtotGun);
