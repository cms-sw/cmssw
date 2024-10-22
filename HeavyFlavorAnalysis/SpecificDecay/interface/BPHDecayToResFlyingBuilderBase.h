#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilderBase_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilderBase_h
/** \class BPHDecayToResFlyingBuilderBase
 *
 *  Description: 
 *     Base class to build a particle decaying to a particle, decaying itself
 *     in cascade, and an additional flying particle
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToFlyingCascadeBuilderBase.h"

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

class BPHDecayToResFlyingBuilderBase : public virtual BPHDecaySpecificBuilderBase,
                                       public virtual BPHDecayConstrainedBuilderBase,
                                       public virtual BPHDecayToFlyingCascadeBuilderBase {
public:
  /** Constructor
   */
  BPHDecayToResFlyingBuilderBase(const BPHEventSetupWrapper& es,
                                 const std::string& resName,
                                 double resMass,
                                 double resWidth,
                                 const std::string& flyName,
                                 double flyMass,
                                 double flyMSigma);

  // deleted copy constructor and assignment operator
  BPHDecayToResFlyingBuilderBase(const BPHDecayToResFlyingBuilderBase& x) = delete;
  BPHDecayToResFlyingBuilderBase& operator=(const BPHDecayToResFlyingBuilderBase& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResFlyingBuilderBase() override = default;

protected:
  BPHDecayToResFlyingBuilderBase();

  /// build candidates
  void fill(BPHRecoBuilder& brb, void* parameters) override;
};

#endif
