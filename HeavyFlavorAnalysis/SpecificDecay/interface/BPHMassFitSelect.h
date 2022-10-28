#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMassFitSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMassFitSelect_h
/** \class BPHMassFitSelect
 *
 *  Description: 
 *     Class for candidate selection by invariant mass (at kinematic fit level)
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassCuts.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassFitSelect : public BPHFitSelect, public BPHMassCuts {
public:
  /** Constructor
   */
  BPHMassFitSelect(double minMass, double maxMass) : BPHMassCuts(minMass, maxMass) { setFitConstraint(); }

  BPHMassFitSelect(const std::string& name, double mass, double sigma, double minMass, double maxMass)
      : BPHMassCuts(minMass, maxMass) {
    setFitConstraint(name, mass, sigma);
  }

  BPHMassFitSelect(const std::string& name, double mass, double minMass, double maxMass)
      : BPHMassCuts(minMass, maxMass) {
    setFitConstraint(name, mass);
  }

  BPHMassFitSelect(const std::string& name, KinematicConstraint* c, double minMass, double maxMass)
      : BPHMassCuts(minMass, maxMass) {
    setFitConstraint(name, c);
  }

  BPHMassFitSelect(const std::string& name, MultiTrackKinematicConstraint* c, double minMass, double maxMass)
      : BPHMassCuts(minMass, maxMass) {
    setFitConstraint(name, c);
  }

  // deleted copy constructor and assignment operator
  BPHMassFitSelect(const BPHMassFitSelect& x) = delete;
  BPHMassFitSelect& operator=(const BPHMassFitSelect& x) = delete;

  /** Destructor
   */
  ~BPHMassFitSelect() override = default;

  /** Operations
   */
  /// select particle
  bool accept(const BPHKinematicFit& cand) const override {
    switch (type) {
      default:
      case none:
        break;
      case mcss:
        cand.kinematicTree(cName, cMass, cSigma);
        break;
      case mcst:
        cand.kinematicTree(cName, cMass);
        break;
      case kf:
        cand.kinematicTree(cName, kc);
        break;
      case mtkf:
        cand.kinematicTree(cName, mtkc);
        break;
    }
    double mass = cand.p4().mass();
    return ((mass >= mMin) && (mass <= mMax));
  }

  /// set fit constraint
  void setFitConstraint() {
    type = none;
    cName = "";
    cMass = -1.0;
    cSigma = -1.0;
    kc = nullptr;
    mtkc = nullptr;
  }
  void setFitConstraint(const std::string& name, double mass) {
    type = mcst;
    cName = name;
    cMass = mass;
    cSigma = -1.0;
    kc = nullptr;
    mtkc = nullptr;
  }
  void setFitConstraint(const std::string& name, double mass, double sigma) {
    type = mcss;
    cName = name;
    cMass = mass;
    cSigma = sigma;
    kc = nullptr;
    mtkc = nullptr;
  }
  void setFitConstraint(const std::string& name, KinematicConstraint* c) {
    type = kf;
    cName = name;
    cMass = -1.0;
    cSigma = -1.0;
    kc = c;
    mtkc = nullptr;
  }
  void setFitConstraint(const std::string& name, MultiTrackKinematicConstraint* c) {
    type = mtkf;
    cName = name;
    cMass = -1.0;
    cSigma = -1.0;
    kc = nullptr;
    mtkc = c;
  }

  /// get fit constraint
  const std::string& getConstrainedName() const { return cName; }
  double getMass() const { return cMass; }
  double getSigma() const { return cSigma; }
  KinematicConstraint* getKC() const { return kc; }
  MultiTrackKinematicConstraint* getMultiTrackKC() const { return mtkc; }

private:
  enum fit_type { none, mcss, mcst, kf, mtkf };

  fit_type type;
  std::string cName;
  double cMass;
  double cSigma;
  KinematicConstraint* kc;
  MultiTrackKinematicConstraint* mtkc;
};

#endif
