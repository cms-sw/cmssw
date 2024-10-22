//-*-c++-*-
//-*-Event.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#ifndef EVENT_HH
#define EVENT_HH

#include "GeneratorInterface/ExhumeInterface/interface/Weight.h"
#include "GeneratorInterface/ExhumeInterface/interface/CrossSection.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace Exhume {

  class Event : public Weight {
  public:
    Event(CrossSection &, CLHEP::HepRandomEngine *);
    ~Event() override;

    inline void SetRandomEngine(CLHEP::HepRandomEngine *engine) {
      randomEngine = engine;
      Process->SetRandomEngine(engine);
    }

    void Generate();
    inline void Setx1Max(const double &xx_) {
      x1Max = xx_;
      return;
    };
    inline void Setx2Max(const double &xx_) {
      x2Max = xx_;
      return;
    };
    inline void Sett1Max(const double &xx_) {
      t1Max = xx_;
      return;
    };
    inline void Sett2Max(const double &xx_) {
      t2Max = xx_;
      return;
    };
    inline void Sett1Min(const double &xx_) {
      t1Min = xx_;
      return;
    };
    inline void Sett2Min(const double &xx_) {
      t2Min = xx_;
      return;
    };
    inline void SetMassRange(const double &Min_, const double &Max_) {
      MinMass = Min_;
      MaxMass = Max_;
      return;
    };

    inline unsigned int GetLastSeed() { return (rand()); };

    inline std::vector<std::pair<double, double> > GetVar() { return (Var); };

    void SetParameterSpace();

    double CrossSectionCalculation();

    inline double GetEfficiency() { return (100.0 * NumberOfEvents / TotalAttempts); };

  private:
    void SelectValues();
    double WeightFunc(const double &) override;

    std::vector<std::pair<double, double> > Var;

    double CSi, CSMass, Sigmai, wgt, yRange;
    double TwoPI, B, InvB, InvBlnB, Root_s, InvRoot_s;
    double SqrtsHat, Eta, t1, t2, Phi1, Phi2, VonNeu;
    double ymax, ymin;
    CrossSection *Process;
    unsigned int NumberOfEvents, TotalAttempts;

    double x1Max, x2Max, t1Min, t1Max, t2Min, t2Max, MinMass, MaxMass;
    double tt1max, tt1min, tt2max, tt2min;

    CLHEP::HepRandomEngine *randomEngine;
  };

}  // namespace Exhume

#endif
