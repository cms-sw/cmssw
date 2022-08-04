#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBsToJPsiPhiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBsToJPsiPhiBuilder_h
/** \class BPHBsToJPsiPhiBuilder
 *
 *  Description: 
 *     Class to build Bs to JPsi Phi candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResResBuilder.h"

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

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHBsToJPsiPhiBuilder
    : public BPHDecayToResResBuilder<BPHRecoCandidate, BPHPlusMinusCandidate, BPHPlusMinusCandidate> {
public:
  /** Constructor
   */
  BPHBsToJPsiPhiBuilder(const BPHEventSetupWrapper& es,
                        const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                        const std::vector<BPHPlusMinusConstCandPtr>& phiCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase("JPsi", BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth),
        BPHDecayToResResBuilder(jpsiCollection, "Phi", phiCollection) {
    setRes1MassRange(2.80, 3.40);
    setRes2MassRange(1.005, 1.035);
    setMassRange(3.50, 8.00);
    setProbMin(0.02);
    setMassFitRange(5.00, 6.00);
    setConstr(true);
  }

  // deleted copy constructor and assignment operator
  BPHBsToJPsiPhiBuilder(const BPHBsToJPsiPhiBuilder& x) = delete;
  BPHBsToJPsiPhiBuilder& operator=(const BPHBsToJPsiPhiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBsToJPsiPhiBuilder() override = default;

  /** Operations
   */
  /// set cuts
  void setJPsiMassMin(double m) { setRes1MassMin(m); }
  void setJPsiMassMax(double m) { setRes1MassMax(m); }
  void setPhiMassMin(double m) { setRes2MassMin(m); }
  void setPhiMassMax(double m) { setRes2MassMax(m); }

  /// get current cuts
  double getJPsiMassMin() const { return getRes1MassMin(); }
  double getJPsiMassMax() const { return getRes1MassMax(); }
  double getPhiMassMin() const { return getRes2MassMin(); }
  double getPhiMassMax() const { return getRes2MassMax(); }

  /// setup parameters for BPHRecoBuilder
  void setup(void* parameters) override {}
};

#endif
