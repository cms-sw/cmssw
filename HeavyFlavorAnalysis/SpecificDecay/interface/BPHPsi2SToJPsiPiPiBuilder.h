#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHPsi2SToJPsiPiPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHPsi2SToJPsiPiPiBuilder_h
/** \class BPHPsi2SToJPsiPiPiBuilder
 *
 *  Description: 
 *     Class to build X3872 to JPsi pi+ pi- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToJPsiPiPiBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <vector>
//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHPsi2SToJPsiPiPiBuilder : public BPHDecayToJPsiPiPiBuilder {
public:
  /** Constructor
   */
  BPHPsi2SToJPsiPiPiBuilder(const BPHEventSetupWrapper& es,
                            const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                            const BPHRecoBuilder::BPHGenericCollection* posCollection,
                            const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayToJPsiPiPiBuilder(jpsiCollection, posCollection, negCollection) {
    setResMassRange(2.80, 3.40);
    setTrkPtMin(1.0);
    setTrkEtaMax(10.0);
    setMassRange(3.00, 4.50);
    setProbMin(0.02);
    setMassFitRange(3.50, 3.80);
  }

  // deleted copy constructor and assignment operator
  BPHPsi2SToJPsiPiPiBuilder(const BPHPsi2SToJPsiPiPiBuilder& x) = delete;
  BPHPsi2SToJPsiPiPiBuilder& operator=(const BPHPsi2SToJPsiPiPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHPsi2SToJPsiPiPiBuilder() override = default;
};

#endif
