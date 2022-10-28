#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilder_h
/** \class BPHDecayToResTrkBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a particle, decaying itself
 *     in cascade, and an additional track, for generic particle types
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResTrkBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;
class BPHParticleNeutralVeto;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class ResType>
class BPHDecayToResTrkBuilder : public BPHDecayToResTrkBuilderBase,
                                public BPHDecayConstrainedBuilder<ProdType, ResType>,
                                public BPHDecaySpecificBuilder<ProdType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  using typename BPHDecayConstrainedBuilder<ProdType, ResType>::res_ptr;

  /** Constructor
   */
  BPHDecayToResTrkBuilder(const BPHEventSetupWrapper& es,
                          const std::string& resName,
                          double resMass,
                          double resWidth,
                          const std::vector<res_ptr>& resCollection,
                          const std::string& trkName,
                          double trkMass,
                          double trkSigma,
                          const BPHRecoBuilder::BPHGenericCollection* trkCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase(resName, resMass, resWidth),
        BPHDecayToResTrkBuilderBase(trkName, trkMass, trkSigma, trkCollection),
        BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection) {}

  // deleted copy constructor and assignment operator
  BPHDecayToResTrkBuilder(const BPHDecayToResTrkBuilder& x) = delete;
  BPHDecayToResTrkBuilder& operator=(const BPHDecayToResTrkBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResTrkBuilder() override = default;

protected:
  BPHDecayToResTrkBuilder(const std::vector<res_ptr>& resCollection,
                          const std::string& trkName,
                          double trkMass,
                          double trkSigma,
                          const BPHRecoBuilder::BPHGenericCollection* trkCollection)
      : BPHDecayToResTrkBuilderBase(trkName, trkMass, trkSigma, trkCollection),
        BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection) {}
};

#endif
