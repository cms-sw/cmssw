
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8EGun : public Py8GunBase {
  public:
    Py8EGun(edm::ParameterSet const&);
    ~Py8EGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // EGun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinE;
    double fMaxE;
    bool fAddAntiParticle;
  };

  // implementation
  //
  Py8EGun::Py8EGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinE = pgun_params.getParameter<double>("MinE");                                      // ,  0.);
    fMaxE = pgun_params.getParameter<double>("MaxE");                                      // ,  0.);
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");                  //, false) ;
  }

  bool Py8EGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();

    for (size_t i = 0; i < fPartIDs.size(); i++) {
      int particleID = fPartIDs[i];  // this is PDG - need to convert to Py8 ???

      double phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
      double ee = (fMaxE - fMinE) * randomEngine().flat() + fMinE;
      double eta = (fMaxEta - fMinEta) * randomEngine().flat() + fMinEta;
      double the = 2. * atan(exp(-eta));

      double mass = (fMasterGen->particleData).m0(particleID);

      double pp = sqrt(ee * ee - mass * mass);
      double px = pp * sin(the) * cos(phi);
      double py = pp * sin(the) * sin(phi);
      double pz = pp * cos(the);

      if (!((fMasterGen->particleData).isParticle(particleID))) {
        particleID = std::fabs(particleID);
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

  const char* Py8EGun::classname() const { return "Py8EGun"; }

  typedef edm::GeneratorFilter<gen::Py8EGun, gen::ExternalDecayDriver> Pythia8EGun;

}  // namespace gen

using gen::Pythia8EGun;
DEFINE_FWK_MODULE(Pythia8EGun);
