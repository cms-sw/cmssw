#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilderBase_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilderBase_h
/** \class BPHDecayToResResBuilderBase
 *
 *  Description: 
 *     Class to build a particle decaying to two particles, decaying
 *     themselves in cascade
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

class BPHEventSetupWrapper;
class BPHRecoBuilder;

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToResResBuilderBase : public virtual BPHDecaySpecificBuilderBase,
                                    public virtual BPHDecayConstrainedBuilderBase {
public:
  /** Constructor
   */
  BPHDecayToResResBuilderBase(const BPHEventSetupWrapper& es,
                              const std::string& res1Name,
                              double res1Mass,
                              double res1Width,
                              const std::string& res2Name);

  // deleted copy constructor and assignment operator
  BPHDecayToResResBuilderBase(const BPHDecayToResResBuilderBase& x) = delete;
  BPHDecayToResResBuilderBase& operator=(const BPHDecayToResResBuilderBase& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResResBuilderBase() override;

  /** Operations
   */

  /// set cuts
  void setRes1MassMin(double m) { setResMassMin(m); }
  void setRes1MassMax(double m) { setResMassMax(m); }
  void setRes1MassRange(double mMin, double mMax) { setResMassRange(mMin, mMax); }
  void setRes2MassMin(double m);
  void setRes2MassMax(double m);
  void setRes2MassRange(double mMin, double mMax);

  /// get current cuts
  double getRes1MassMin() const { return getResMassMin(); }
  double getRes1MassMax() const { return getResMassMax(); }
  double getRes2MassMin() const { return res2Sel->getMassMin(); }
  double getRes2MassMax() const { return res2Sel->getMassMax(); }

protected:
  BPHDecayToResResBuilderBase(const std::string& res2Name);

  std::string sName;

  BPHMassSelect* res2Sel;

  virtual void addRes1Collection(BPHRecoBuilder& brb) { addResCollection(brb); }
  virtual void addRes2Collection(BPHRecoBuilder& brb) = 0;

  /// build candidates
  void fill(BPHRecoBuilder& brb, void* parameters) override;

  class DZSelect : public BPHVertexSelect {
  public:
    DZSelect(const std::string* n) : name(n) {}
    bool accept(const BPHDecayVertex& cand) const override { return true; }
    bool accept(const BPHDecayVertex& cand, const BPHRecoBuilder* builder) const override {
      const auto& res1 = builder->getComp(*name);
      return (fabs(cand.vertex().z() - res1->vertex().z()) < 1.0);
    }

  private:
    const std::string* name;
  };
  DZSelect dzFilter;
};

#endif
