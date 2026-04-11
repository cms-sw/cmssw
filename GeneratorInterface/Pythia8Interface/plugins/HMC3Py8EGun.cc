
#include <memory>

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8HMC3GunBase.h"

// EvtGen plugin
//
#include "Pythia8Plugins/EvtGen.h"

namespace gen {

  class HMC3Py8EGun : public Py8HMC3GunBase {
  public:
    HMC3Py8EGun(edm::ParameterSet const&);
    ~HMC3Py8EGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;
    void evtGenDecay();

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
  HMC3Py8EGun::HMC3Py8EGun(edm::ParameterSet const& ps) : Py8HMC3GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinE = pgun_params.getParameter<double>("MinE");                                      // ,  0.);
    fMaxE = pgun_params.getParameter<double>("MaxE");                                      // ,  0.);
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");                  //, false) ;

    if (useEvtGen) {
      edm::LogInfo("Pythia8Interface") << "Creating and initializing pythia8 EvtGen plugin";
      evtgenDecays = std::make_shared<Pythia8::EvtGenDecays>(fMasterGen.get(), evtgenDecFile, evtgenPdlFile);
      for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
        evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
    }
  }

  bool HMC3Py8EGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();

    int NTotParticles = fPartIDs.size();
    if (fAddAntiParticle)
      NTotParticles *= 2;

    // energy below is dummy, it is not used
    (fMasterGen->event).append(990, -11, 0, 0, 2, 1 + NTotParticles, 0, 0, 0., 0., 0., 15000., 15000.);

    int colorindex = 101;

    for (size_t i = 0; i < fPartIDs.size(); i++) {
      int particleID = fPartIDs[i];  // this is PDG - need to convert to Py8 ???
      if ((std::abs(particleID) <= 6 || particleID == 21) && !(fAddAntiParticle)) {
        throw cms::Exception("PythiaError") << "Attempting to generate quarks or gluons without setting "
                                               "AddAntiParticle to true. This will not handle color properly."
                                            << std::endl;
      }

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
      if (1 <= std::abs(particleID) && std::abs(particleID) <= 6) {  // quarks
        (fMasterGen->event).append(particleID, 23, 1, 0, 0, 0, colorindex, 0, px, py, pz, ee, mass);
        if (!fAddAntiParticle)
          colorindex += 1;
      } else if (std::abs(particleID) == 21) {  // gluons
        (fMasterGen->event).append(21, 23, 1, 0, 0, 0, colorindex, colorindex + 1, px, py, pz, ee, mass);
        if (!fAddAntiParticle) {
          colorindex += 2;
        }
      }
      // other
      else {
        (fMasterGen->event).append(particleID, 1, 1, 0, 0, 0, 0, 0, px, py, pz, ee, mass);
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
          (fMasterGen->event).append(-particleID, 23, 1, 0, 0, 0, 0, colorindex, -px, -py, -pz, ee, mass);
          colorindex += 1;
        } else if (std::abs(particleID) == 21) {  // gluons
          (fMasterGen->event).append(21, 23, 1, 0, 0, 0, colorindex + 1, colorindex, -px, -py, -pz, ee, mass);
          colorindex += 2;
        } else {
          if ((fMasterGen->particleData).isParticle(-particleID)) {
            (fMasterGen->event).append(-particleID, 1, 1, 0, 0, 0, 0, 0, -px, -py, -pz, ee, mass);
          } else {
            (fMasterGen->event).append(particleID, 1, 1, 0, 0, 0, 0, 0, -px, -py, -pz, ee, mass);
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

    event3() = std::make_unique<HepMC3::GenEvent>();
    return toHepMC.fill_next_event(fMasterGen->event, event3().get());
  }

  void HMC3Py8EGun::evtGenDecay() {
    if (evtgenDecays.get())
      evtgenDecays->decay();
  }

  const char* HMC3Py8EGun::classname() const { return "HMC3Py8EGun"; }

  typedef edm::GeneratorFilter<gen::HMC3Py8EGun, gen::ExternalDecayDriver> Pythia8HepMC3EGun;

}  // namespace gen

using gen::Pythia8HepMC3EGun;
DEFINE_FWK_MODULE(Pythia8HepMC3EGun);
