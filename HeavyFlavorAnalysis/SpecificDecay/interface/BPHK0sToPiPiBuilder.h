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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

#include "FWCore/Framework/interface/Event.h"

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
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const BPHRecoBuilder::BPHGenericCollection* posCollection,
                      const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayToV0SameMassBuilder(es,
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
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                      const std::string& searchList = "cfp")
      : BPHDecayToV0SameMassBuilder(es,
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
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                      const std::string& searchList = "cfp")
      : BPHDecayToV0SameMassBuilder(es,
                                    "PionPos",
                                    "PionNeg",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    vpCollection,
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
  ~BPHK0sToPiPiBuilder() override {}
};

#endif
