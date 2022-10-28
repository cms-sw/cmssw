#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToFlyingCascadeBuilderBase_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToFlyingCascadeBuilderBase_h
/** \class BPHDecayToFlyingCascadeBuilderBase
 *
 *  Description: 
 *     Base class to build a particle having a flying particle in the
 *     final state
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKinFitChi2Select.h"

class BPHEventSetupWrapper;
class BPHRecoBuilder;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToFlyingCascadeBuilderBase : public virtual BPHDecaySpecificBuilderBase,
                                           public virtual BPHDecayGenericBuilderBase {
public:
  /** Constructor
   */
  BPHDecayToFlyingCascadeBuilderBase(const BPHEventSetupWrapper& es,
                                     const std::string& flyName,
                                     double flyMass,
                                     double flyMSigma);

  // deleted copy constructor and assignment operator
  BPHDecayToFlyingCascadeBuilderBase(const BPHDecayToFlyingCascadeBuilderBase& x) = delete;
  BPHDecayToFlyingCascadeBuilderBase& operator=(const BPHDecayToFlyingCascadeBuilderBase& x) = delete;

  /** Destructor
   */
  ~BPHDecayToFlyingCascadeBuilderBase() override;

  /** Operations
   */
  /// get original daughters map
  const std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>& daughMap() const { return dMap; }

  /// set cuts
  void setFlyingMassMin(double m);
  void setFlyingMassMax(double m);
  void setFlyingMassRange(double mMin, double mMax);
  void setKinFitProbMin(double p);

  /// get current cuts
  double getFlyingMassMin() const { return flySel->getMassMin(); }
  double getFlyingMassMax() const { return flySel->getMassMax(); }
  double getKinFitProbMin() const { return kfChi2Sel->getProbMin(); }

protected:
  BPHDecayToFlyingCascadeBuilderBase(const std::string& flyName, double flyMass, double flyMSigma);
  BPHDecayToFlyingCascadeBuilderBase();

  std::string fName;
  double fMass;
  double fMSigma;

  BPHMassFitSelect* flySel;
  BPHKinFitChi2Select* kfChi2Sel;

  std::map<const BPHRecoCandidate*, const BPHRecoCandidate*> dMap;

  virtual void addFlyCollection(BPHRecoBuilder& brb) = 0;
};

#endif
