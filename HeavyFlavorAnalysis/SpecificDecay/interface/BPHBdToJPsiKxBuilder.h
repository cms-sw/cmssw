#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKxBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKxBuilder_h
/** \class BPHBdToJPsiKxBuilder
 *
 *  Description: 
 *     Class to build B0 to JPsi K*0 candidates
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

class BPHBdToJPsiKxBuilder
    : public BPHDecayToResResBuilder<BPHRecoCandidate, BPHPlusMinusCandidate, BPHPlusMinusCandidate> {
public:
  /** Constructor
   */
  BPHBdToJPsiKxBuilder(const BPHEventSetupWrapper& es,
                       const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                       const std::vector<BPHPlusMinusConstCandPtr>& kx0Collection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase("JPsi", BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth),
        BPHDecayToResResBuilder(jpsiCollection, "Kx0", kx0Collection) {
    setRes1MassRange(2.80, 3.40);
    setRes2MassRange(0.80, 1.00);
    setMassRange(3.50, 8.00);
    setProbMin(0.02);
    setMassFitRange(5.00, 6.00);
    setConstr(true);
  }

  // deleted copy constructor and assignment operator
  BPHBdToJPsiKxBuilder(const BPHBdToJPsiKxBuilder& x) = delete;
  BPHBdToJPsiKxBuilder& operator=(const BPHBdToJPsiKxBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBdToJPsiKxBuilder() override = default;

  /** Operations
   */
  /// set cuts
  void setJPsiMassMin(double m) { setRes1MassMin(m); }
  void setJPsiMassMax(double m) { setRes1MassMax(m); }
  void setKxMassMin(double m) { setRes2MassMin(m); }
  void setKxMassMax(double m) { setRes2MassMax(m); }

  /// get current cuts
  double getJPsiMassMin() const { return getRes1MassMin(); }
  double getJPsiMassMax() const { return getRes1MassMax(); }
  double getKxMassMin() const { return getRes2MassMin(); }
  double getKxMassMax() const { return getRes2MassMax(); }

  /// setup parameters for BPHRecoBuilder
  void setup(void* parameters) override {}
};

#endif
