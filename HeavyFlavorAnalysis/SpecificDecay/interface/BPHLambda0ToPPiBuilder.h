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

class BPHLambda0ToPPiBuilder : public BPHDecayToV0DiffMassBuilder {
public:
  /** Constructor
   */
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const BPHRecoBuilder::BPHGenericCollection* protonCollection,
                         const BPHRecoBuilder::BPHGenericCollection* pionCollection)
      : BPHDecayToV0DiffMassBuilder(es,
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
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                         const std::string& searchList = "cfp")
      : BPHDecayToV0DiffMassBuilder(es,
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
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                         const std::string& searchList = "cfp")
      : BPHDecayToV0DiffMassBuilder(es,
                                    "Proton",
                                    BPHParticleMasses::protonMass,
                                    BPHParticleMasses::protonMSigma,
                                    "Pion",
                                    BPHParticleMasses::pionMass,
                                    BPHParticleMasses::pionMSigma,
                                    vpCollection,
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
  ~BPHLambda0ToPPiBuilder() override {}
};

#endif
