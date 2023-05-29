#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilder_h
/** \class BPHDecayToResFlyingBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a particle, decaying itself
 *     in cascade, and an additional flying particle, for generic particle types
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResFlyingBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToFlyingCascadeBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <iostream>
//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class ResType, class FlyingType>
class BPHDecayToResFlyingBuilder : public BPHDecayToResFlyingBuilderBase,
                                   public BPHDecayConstrainedBuilder<ProdType, ResType>,
                                   public BPHDecayToFlyingCascadeBuilder<ProdType, FlyingType>,
                                   public BPHDecaySpecificBuilder<ProdType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  using typename BPHDecayConstrainedBuilder<ProdType, ResType>::res_ptr;
  using typename BPHDecayToFlyingCascadeBuilder<ProdType, FlyingType>::flying_ptr;

  /** Constructor
   */
  BPHDecayToResFlyingBuilder(const BPHEventSetupWrapper& es,
                             const std::string& resName,
                             double resMass,
                             double resWidth,
                             const std::vector<res_ptr>& resCollection,
                             const std::string& flyName,
                             double flyMass,
                             double flyMSigma,
                             const std::vector<flying_ptr>& flyCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase(resName, resMass, resWidth),
        BPHDecayToFlyingCascadeBuilderBase(flyName, flyMass, flyMSigma),
        BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection),
        BPHDecayToFlyingCascadeBuilder<ProdType, FlyingType>(flyCollection) {}

  // deleted copy constructor and assignment operator
  BPHDecayToResFlyingBuilder(const BPHDecayToResFlyingBuilder& x) = delete;
  BPHDecayToResFlyingBuilder& operator=(const BPHDecayToResFlyingBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResFlyingBuilder() override = default;

protected:
  BPHDecayToResFlyingBuilder(const std::vector<res_ptr>& resCollection, const std::vector<flying_ptr>& flyCollection)
      : BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection),
        BPHDecayToFlyingCascadeBuilder<ProdType, FlyingType>(flyCollection) {}

  void fillRecList() override {
    BPHDecaySpecificBuilder<ProdType>::fillRecList();
    this->fitAndFilter(this->recList);
    return;
  }
};

#endif
