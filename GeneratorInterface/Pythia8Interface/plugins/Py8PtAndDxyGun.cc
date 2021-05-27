#include <memory>

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

namespace gen {

  class Py8PtAndDxyGun : public Py8GunBase {
  public:
    Py8PtAndDxyGun(edm::ParameterSet const&);
    ~Py8PtAndDxyGun() override {}

    bool generatePartonsAndHadronize() override;
    const char* classname() const override;

  private:
    // PtAndDxyGun particle(s) characteristics
    double fMinEta;
    double fMaxEta;
    double fMinPt;
    double fMaxPt;
    bool fAddAntiParticle;
    double fDxyMin;
    double fDxyMax;
    double fLxyMax;
    double fLzMax;
    double fConeRadius;
    double fConeH;
    double fDistanceToAPEX;
  };

  // implementation
  //
  Py8PtAndDxyGun::Py8PtAndDxyGun(edm::ParameterSet const& ps) : Py8GunBase(ps) {
    // ParameterSet defpset ;
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");  // , defpset ) ;
    fMinEta = pgun_params.getParameter<double>("MinEta");                                  // ,-2.2);
    fMaxEta = pgun_params.getParameter<double>("MaxEta");                                  // , 2.2);
    fMinPt = pgun_params.getParameter<double>("MinPt");                                    // ,  0.);
    fMaxPt = pgun_params.getParameter<double>("MaxPt");                                    // ,  0.);
    fAddAntiParticle = pgun_params.getParameter<bool>("AddAntiParticle");                  //, false) ;
    fDxyMin = pgun_params.getParameter<double>("dxyMin");
    fDxyMax = pgun_params.getParameter<double>("dxyMax");
    fLxyMax = pgun_params.getParameter<double>("LxyMax");
    fLzMax = pgun_params.getParameter<double>("LzMax");
    fConeRadius = pgun_params.getParameter<double>("ConeRadius");
    fConeH = pgun_params.getParameter<double>("ConeH");
    fDistanceToAPEX = pgun_params.getParameter<double>("DistanceToAPEX");
  }

  bool Py8PtAndDxyGun::generatePartonsAndHadronize() {
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
        bool passLxy = false;
        bool passLz = false;

        phi = (fMaxPhi - fMinPhi) * randomEngine().flat() + fMinPhi;
        dxy = (fDxyMax - fDxyMin) * randomEngine().flat() + fDxyMin;
        float dxysign = randomEngine().flat() - 0.5;
        if (dxysign < 0)
          dxy = -dxy;

        pt = (fMaxPt - fMinPt) * randomEngine().flat() + fMinPt;
        px = pt * cos(phi);
        py = pt * sin(phi);

        for (int i = 0; i < 10000; i++) {
          vx = 2 * fLxyMax * randomEngine().flat() - fLxyMax;
          vy = (pt * dxy + vx * py) / px;
          lxy = sqrt(vx * vx + vy * vy);
          if (lxy < std::abs(fLxyMax) && (vx * px + vy * py) > 0) {
            passLxy = true;
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
        if (pz < 0)
          vz = -vz;
        passLoop = (passLxy && passLz);

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
        (fMasterGen->event).back().vProd(-vx, -vy, -vz, time);
      }
    }

    if (!fMasterGen->next())
      return false;
    evtGenDecay();

    event() = std::make_unique<HepMC::GenEvent>();
    return toHepMC.fill_next_event(fMasterGen->event, event().get());
  }

  const char* Py8PtAndDxyGun::classname() const { return "Py8PtAndDxyGun"; }

  typedef edm::GeneratorFilter<gen::Py8PtAndDxyGun, gen::ExternalDecayDriver> Pythia8PtAndDxyGun;

}  // namespace gen

using gen::Pythia8PtAndDxyGun;
DEFINE_FWK_MODULE(Pythia8PtAndDxyGun);
