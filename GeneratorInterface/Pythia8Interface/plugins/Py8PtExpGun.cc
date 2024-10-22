#include <memory>

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8PtExpGun : public Py8GunBase {
  public:
    Py8PtExpGun(edm::ParameterSet const&);
    ~Py8PtExpGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // PtExpGun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinPt;
    double fMaxPt;
    bool fAddAntiParticle;
  };

  // implementation
  //
  Py8PtExpGun::Py8PtExpGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinPt = pgun_params.getParameter<double>("MinPt");                                    // ,  0.);
    fMaxPt = pgun_params.getParameter<double>("MaxPt");                                    // ,  0.);
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");                  //, false) ;
  }

  bool Py8PtExpGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();

    int NTotParticles = fPartIDs.size();
    if (fAddAntiParticle)
      NTotParticles *= 2;

    // energy below is dummy, it is not used
    (fMasterGen->event).append(990, -11, 0, 0, 2, 1 + NTotParticles, 0, 0, 0., 0., 0., 15000., 15000.);

    for (size_t i = 0; i < fPartIDs.size(); i++) {
      int particleID = fPartIDs[i];  // this is PDG - need to convert to Py8 ???

      double phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
      double eta = (fMaxEta - fMinEta) * randomEngine().flat() + fMinEta;
      double the = 2. * atan(exp(-eta));

      //-log(flat) = exponential distribution
      //  need the /10.0 and the min with 1.0 to make sure pt doesn't go too high
      //      10.0 chosen to give last pt bin (overflow bin) a reasonable (not unnaturally high) content
      double pt = (std::min(-1 / 10.0 * log(randomEngine().flat()), 1.0)) * (fMaxPt - fMinPt) + fMinPt;

      double mass = (fMasterGen->particleData).m0(particleID);

      double pp = pt / sin(the);  // sqrt( ee*ee - mass*mass );
      double ee = sqrt(pp * pp + mass * mass);

      double px = pt * cos(phi);
      double py = pt * sin(phi);
      double pz = pp * cos(the);

      if (!((fMasterGen->particleData).isParticle(particleID))) {
        particleID = std::abs(particleID);
      }
      if (1 <= std::abs(particleID) && std::abs(particleID) <= 6)  // quarks
        (fMasterGen->event).append(particleID, 23, 1, 0, 0, 0, 101, 0, px, py, pz, ee, mass);
      else if (std::abs(particleID) == 21)  // gluons
        (fMasterGen->event).append(21, 23, 1, 0, 0, 0, 101, 102, px, py, pz, ee, mass);
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
          (fMasterGen->event).append(-particleID, 23, 1, 0, 0, 0, 0, 101, -px, -py, -pz, ee, mass);
        } else if (std::abs(particleID) == 21) {  // gluons
          (fMasterGen->event).append(21, 23, 1, 0, 0, 0, 102, 101, -px, -py, -pz, ee, mass);
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

    event() = std::make_unique<HepMC::GenEvent>();
    return toHepMC.fill_next_event(fMasterGen->event, event().get());
  }

  const char* Py8PtExpGun::classname() const { return "Py8PtExpGun"; }

  typedef edm::GeneratorFilter<gen::Py8PtExpGun, gen::ExternalDecayDriver> Pythia8PtExpGun;

}  // namespace gen

using gen::Pythia8PtExpGun;
DEFINE_FWK_MODULE(Pythia8PtExpGun);
