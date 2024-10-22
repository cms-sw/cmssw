//-*-C++-*-
//-*-CrossSection.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////

#ifndef CROSSSECTION_HH
#define CROSSSECTION_HH

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "GeneratorInterface/ExhumeInterface/interface/I.h"
//#include "GeneratorInterface/ExhumeInterface/interface/PythiaRecord.h"
#include "GeneratorInterface/ExhumeInterface/interface/Particle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"
//#include "CLHEP/HepMC/ConvertHEPEVT.h"
//#include "CLHEP/HepMC/CBhepevt.h"

namespace CLHEP {
  class HepRandomEngine;
}

/////////////////////////////////////////////////////////////////////////////
namespace Exhume {

  typedef std::pair<double, double> fEntry;
  typedef std::pair<const char *, char *> PCharPair;
  typedef std::pair<const char *, const void *> PConstVoidPair;

  class CrossSection {
  public:
    CrossSection(const edm::ParameterSet &);
    virtual ~CrossSection();
    virtual void MaximiseSubParameters() = 0;
    virtual void SetPartons() = 0;
    virtual void SetSubParameters() = 0;
    virtual double SubParameterRange() = 0;
    virtual double SubParameterWeight() = 0;

    double AlphaS(const double &);

    inline void SetRandomEngine(CLHEP::HepRandomEngine *engine) { randomEngine = engine; }

    inline double GetRg(const double &x_, const double &Qt) { return (Rg_(x_, Qt)); };

    //.....
    inline double Differential() {
      if (x1 > 1.0 || x2 > 1.0) {
        return (0.0);
      }

      //return( SubProcess() );

      //Factor of 2/sqrt(sHat) because we integrate dM not dln(M^2)
      return (2.0 * InvSqrtsHat * Lumi() * SubProcess() * exp(B * (t1 + t2)));
    };
    //.....

    inline double GetB() { return (B); };

    inline double GetRoot_s() { return (root_s); };

    inline std::map<double, double> Getfg2Map() { return (fg2Map); };

    inline CLHEP::HepLorentzVector GetProton1() { return (Proton1); };

    inline CLHEP::HepLorentzVector GetProton2() { return (Proton2); };

    inline std::vector<Particle> GetPartons() { return (Partons); };

    inline double Getx1() { return (x1); };

    inline double Getx2() { return (x2); };

    inline double Gett1() { return (t1); };

    inline double Gett2() { return (t2); };

    inline double GetsHat() { return (sHat); };

    inline double GetSqrtsHat() { return (SqrtsHat); };

    inline double GetEta() { return (y); };

    inline double GetPhi1() { return (Phi1); };

    inline double GetPhi2() { return (Phi2); };

    void Hadronise();
    void SetKinematics(const double &, const double &, const double &, const double &, const double &, const double &);

    inline std::string GetName() { return (Name); };

  private:
    void AlphaSInit();

    double Fg1Fg2(const double &, const double &, const double &);
    double Fg1Fg2(const int &, const double &, const double &);

    inline double Fg_Qt2(const double &Qt2_, const double &x_, const double &xp_) {
      double grad = 5.0 * (Txg(1.1 * Qt2_, x_) - Txg(0.9 * Qt2_, x_)) / Qt2_;

      //protect from Qt2 == 0

      if (grad != grad) {
        return (0.0);
      }
      return (grad);
    };

    template <typename T_>
    inline void insert(const std::string _name_, const T_ _x_) {
      PMap.insert(std::pair<std::string, PConstVoidPair>(_name_, PConstVoidPair(typeid(_x_).name(), _x_)));
    }

    virtual double SubProcess() = 0;

    void LumiInit();
    double Lumi();
    double Lumi_();
    void NoMem();
    void ReadCard(const std::string &);
    double Splitting(const double &);
    double T(const double &);
    double TFast(const double &);

    bool TBegin, TInterpolate;
    std::map<double, std::map<double, double> > TMap2d;
    std::map<double, std::map<double, double> >::iterator TMjuHigh, TMjuLow;

    double Txg(const double &, const double &);

    double Rg1Rg2(const double &);
    double Rg_(const double &, double);

    std::map<double, std::map<double, double> > RgMap2d;
    std::map<double, std::map<double, double> >::iterator RgHigh[2], RgLow[2];
    bool RgInterpolate[2], RgBegin;

    double B, Freeze, LambdaQCD, Rg, Survive;

    double PDF;

    double ASFreeze, ASConst;
    //.............................
    //Used for Lumi function:

    double MinQt2, MidQt2, InvMidQt2, InvMidQt4, InvMaxQt2, LumSimpsIncr;
    unsigned int LumAccuracy, LumNStart;
    int LumNSimps, LumNSimps_1;
    double *LumSimpsFunc, *_Qt2, *_Qt, *_KtLow, *_KtHigh, *_AlphaS, *_CfAs_2PIRg, *_NcAs_2PI;
    double LumConst;
    double Inv2PI, Nc_2PI, Cf_2PIRg;

    std::map<std::string, PConstVoidPair> PMap;
    std::map<const char *, char *> TypeMap;
    std::map<double, double> fg2Map;

    //...............................
    //Used for T:
    double TConst;
    int Tn, Tn_1;
    double *TFunc;

    //...............................
    //Used for Splitting:
    double Inv3;

    int Proton1Id, Proton2Id;

  protected:
    std::string Name;

    std::complex<double> F0(const double &);
    std::complex<double> f(const double &);
    std::complex<double> Fsf(const double &);

    //PPhi is azimuthal angle between protons.
    //InvSqrtsHat = 1/sHat
    //Others are obvious.
    double x1, x1p, x2, x2p, t1, t2, sHat, SqrtsHat, sHat2, InvsHat, InvsHat2, InvSqrtsHat, y, PPhi, Phi1, Phi2, Mju2,
        Mju, LnMju2, Pt1, Pt2, Pt1DotPt2, x1x2, ey;

    double AlphaEw, gw, HiggsVev, LambdaW;

    double BottomMass, CharmMass, StrangeMass, TopMass;

    double MuonMass, TauMass;

    double HiggsMass, WMass, ZMass;

    int FNAL_or_LHC;
    double root_s, s, Invs;

    CLHEP::HepLorentzVector CentralVector;
    CLHEP::HepLorentzVector Proton1, Proton2, P1In, P2In;
    /*
    struct Particle {
      CLHEP::HepLorentzVector p;
      int id;
    };
    */

    std::vector<Particle> Partons;
    double Invsx1x2, InvV1MinusV2;

    double Gev2fb;

    std::string lhapdfSetPath_;

    CLHEP::HepRandomEngine *randomEngine;
  };
}  // namespace Exhume

#endif
////////////////////////////////////////////////////////////////////////////
