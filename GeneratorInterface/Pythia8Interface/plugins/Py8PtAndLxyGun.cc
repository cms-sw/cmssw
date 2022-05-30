#include <memory>
#include <algorithm>

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8PtAndLxyGun : public Py8GunBase {
  public:
    Py8PtAndLxyGun(edm::ParameterSet const&);
    ~Py8PtAndLxyGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // PtAndLxyGun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinPt;
    double fMaxPt;
    bool fAddAntiParticle;
    double fDxyMax;
    double fDzMax;
    double fLxyMin;
    double fLxyMax;
    double fLzMax;
    double fConeRadius;
    double fConeH;
    double fDistanceToAPEX;
    double fLxyBackFraction;
    double fLzOppositeFraction;
  };

  // implementation
  //
  Py8PtAndLxyGun::Py8PtAndLxyGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");
    fMinEta = pgun_params.getParameter<double>("MinEta");
    fMaxEta = pgun_params.getParameter<double>("MaxEta");
    fMinPt = pgun_params.getParameter<double>("MinPt");
    fMaxPt = pgun_params.getParameter<double>("MaxPt");
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");
    fDxyMax = pgun_params.getParameter<double>("dxyMax");
    fDzMax = pgun_params.getParameter<double>("dzMax");
    fLxyMin = pgun_params.getParameter<double>("LxyMin");
    fLxyMax = pgun_params.getParameter<double>("LxyMax");
    fLzMax = pgun_params.getParameter<double>("LzMax");
    fConeRadius = pgun_params.getParameter<double>("ConeRadius");
    fConeH = pgun_params.getParameter<double>("ConeH");
    fDistanceToAPEX = pgun_params.getParameter<double>("DistanceToAPEX");
    fLxyBackFraction = std::clamp(pgun_params.getParameter<double>("LxyBackFraction"), 0., 1.);
    fLzOppositeFraction = std::clamp(pgun_params.getParameter<double>("LzOppositeFraction"), 0., 1.);
  }

  bool Py8PtAndLxyGun::generatePartonsAndHadronize() {
    fMasterGen->event.reset();

    for (size_t i = 0; i < fPartIDs.size(); i++) {
      int particleID = fPartIDs[i];  // this is PDG - need to convert to Py8 ???

      double phi = 0;
      double dxy = 0;
      double pt = 0;
      double eta = 0;
      double px = 0;
      double py = 0;
      double pz = 0;
      double mass = 0;
      double ee = 0;
      double vx = 0;
      double vy = 0;
      double vz = 0;
      double lxy = 0;

      bool passLoop = false;
      while (!passLoop) {
        bool passDxy = false;
        bool passLz = false;
        bool passDz = false;

        phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
        pt = (fMaxPt - fMinPt) * randomEngine().flat() + fMinPt;
        px = pt * cos(phi);
        py = pt * sin(phi);

        lxy = (fLxyMax - fLxyMin) * randomEngine().flat() + fLxyMin;

        int sign = 1;
        for (int i = 0; i < 10000; i++) {
          double vphi = 2 * M_PI * randomEngine().flat();
          vx = lxy * cos(vphi);
          vy = lxy * sin(vphi);

          dxy = -vx * sin(phi) + vy * cos(phi);

          sign = 1;
          if (fLxyBackFraction > 0 && randomEngine().flat() <= fLxyBackFraction) {
            sign = -1;
          }
          if ((std::abs(dxy) < fDxyMax || fDxyMax < 0) && sign * (vx * px + vy * py) > 0) {
            passDxy = true;
            break;
          }
        }

        eta = (fMaxEta - fMinEta) * randomEngine().flat() + fMinEta;
        double theta = 2. * atan(exp(-eta));

        mass = (fMasterGen->particleData).m0(particleID);

        double pp = pt / sin(theta);  // sqrt( ee*ee - mass*mass );
        ee = sqrt(pp * pp + mass * mass);

        pz = pp * cos(theta);

        float coneTheta = fConeRadius / fConeH;
        for (int j = 0; j < 100; j++) {
          vz = fLzMax * randomEngine().flat();  // this is abs(vz)
          float v0 = vz - fDistanceToAPEX;
          if (v0 <= 0 || lxy * lxy / (coneTheta * coneTheta) > v0 * v0) {
            passLz = true;
            break;
          }
        }

        if (fLzOppositeFraction > 0 && randomEngine().flat() <= fLzOppositeFraction)
          sign *= -1;
        if (sign * pz < 0)
          vz = -vz;

        double dz = vz - (vx * cos(phi) + vy * sin(phi)) / tan(theta);
        if (std::abs(dz) < fDzMax || fDzMax < 0) {
          passDz = true;
        }

        passLoop = (passDxy && passLz && passDz);
        if (passLoop)
          break;
      }

      float time = sqrt(vx * vx + vy * vy + vz * vz);

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
      (fMasterGen->event).back().vProd(vx, vy, vz, time);

      // Here also need to add anti-particle (if any)
      // otherwise just add a 2nd particle of the same type
      // (for example, gamma).
      // Added anti-particle has momentum opposite to corresponding
      // particle, (px,py,pz)=>(-px,-py,-pz), and production vertex
      // symmetric wrt (0,0,0), (vx, vy, vz)=>(-vx, -vy, -vz).
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
        (fMasterGen->event).back().vProd(-vx, -vy, -vz, time);
      }
    }

    if (!fMasterGen->next())
      return false;
    evtGenDecay();

    event() = std::make_unique<HepMC::GenEvent>();
    return toHepMC.fill_next_event(fMasterGen->event, event().get());
  }

  const char* Py8PtAndLxyGun::classname() const { return "Py8PtAndLxyGun"; }

  typedef edm::GeneratorFilter<gen::Py8PtAndLxyGun, gen::ExternalDecayDriver> Pythia8PtAndLxyGun;

}  // namespace gen

using gen::Pythia8PtAndLxyGun;
DEFINE_FWK_MODULE(Pythia8PtAndLxyGun);
