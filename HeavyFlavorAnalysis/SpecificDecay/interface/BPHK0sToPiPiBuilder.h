#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHK0sToPiPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHK0sToPiPiBuilder_h
/** \class BPHK0sToPiPiBuilder
 *
 *  Description: 
 *     Class to build K0s to pi+ pi- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0SameMassBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
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

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHK0sToPiPiBuilder : public BPHDecayToV0SameMassBuilder {
public:
  /** Constructor
   */
  BPHK0sToPiPiBuilder(const BPHEventSetupWrapper& es,
                      const BPHRecoBuilder::BPHGenericCollection* posCollection,
                      const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayGenericBuilderBase(es),
        BPHDecayToV0SameMassBuilder(es,
                                    "PionPos",
                                    "PionNeg",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    posCollection,
                                    negCollection) {
    setPtMin(0.7);
    setEtaMax(10.0);
    setMassRange(0.40, 0.60);
  }

  template <class V0VertexType>
  BPHK0sToPiPiBuilder(const BPHEventSetupWrapper& es,
                      const std::vector<V0VertexType>* v0Collection,
                      const std::string& searchList = "cfp")
      : BPHDecayGenericBuilderBase(es),
        BPHDecayToV0SameMassBuilder(es,
                                    "PionPos",
                                    "PionNeg",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    v0Collection,
                                    searchList) {
    setPtMin(0.0);
    setEtaMax(10.0);
    setMassRange(0.00, 2.00);
  }

  // deleted copy constructor and assignment operator
  BPHK0sToPiPiBuilder(const BPHK0sToPiPiBuilder& x) = delete;
  BPHK0sToPiPiBuilder& operator=(const BPHK0sToPiPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHK0sToPiPiBuilder() override = default;
};

#endif
