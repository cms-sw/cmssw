#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHLambda0ToPPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHLambda0ToPPiBuilder_h
/** \class BPHLambda0ToPPiBuilder
 *
 *  Description: 
 *     Class to build Lambda0 to p pi candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0DiffMassBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexCompositePtrCandidate.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

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

class BPHLambda0ToPPiBuilder : public BPHDecayToV0DiffMassBuilder {
public:
  /** Constructor
   */
  BPHLambda0ToPPiBuilder(const BPHEventSetupWrapper& es,
                         const BPHRecoBuilder::BPHGenericCollection* protonCollection,
                         const BPHRecoBuilder::BPHGenericCollection* pionCollection)
      : BPHDecayGenericBuilderBase(es),
        BPHDecayToV0DiffMassBuilder(es,
                                    "Proton",
                                    BPHParticleMasses::protonMass,
                                    BPHParticleMasses::protonMSigma,
                                    "Pion",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    protonCollection,
                                    pionCollection,
                                    BPHParticleMasses::lambda0Mass) {
    setPtMin(0.7);
    setEtaMax(10.0);
    setMassRange(0.80, 1.40);
  }

  template <class V0VertexType>
  BPHLambda0ToPPiBuilder(const BPHEventSetupWrapper& es,
                         const std::vector<V0VertexType>* v0Collection,
                         const std::string& searchList = "cfp")
      : BPHDecayGenericBuilderBase(es),
        BPHDecayToV0DiffMassBuilder(es,
                                    "Proton",
                                    BPHParticleMasses::protonMass,
                                    BPHParticleMasses::protonMSigma,
                                    "Pion",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    v0Collection,
                                    BPHParticleMasses::lambda0Mass) {
    setPtMin(0.0);
    setEtaMax(10.0);
    setMassRange(0.00, 3.00);
  }

  // deleted copy constructor and assignment operator
  BPHLambda0ToPPiBuilder(const BPHLambda0ToPPiBuilder& x) = delete;
  BPHLambda0ToPPiBuilder& operator=(const BPHLambda0ToPPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHLambda0ToPPiBuilder() override = default;
};

#endif
