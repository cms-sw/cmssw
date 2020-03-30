#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKsBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKsBuilder_h
/** \class BPHBdToJPsiKsBuilder
 *
 *  Description: 
 *     Class to build B0 to JPsi K0s candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResFlyingBuilder.h"

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

class BPHBdToJPsiKsBuilder : public BPHDecayToResFlyingBuilder {
public:
  /** Constructor
   */
  BPHBdToJPsiKsBuilder(const edm::EventSetup& es,
                       const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                       const std::vector<BPHPlusMinusConstCandPtr>& k0sCollection)
      : BPHDecayToResFlyingBuilder(es,
                                   "JPsi",
                                   BPHParticleMasses::jPsiMass,
                                   BPHParticleMasses::jPsiMWidth,
                                   jpsiCollection,
                                   "K0s",
                                   BPHParticleMasses::k0sMass,
                                   BPHParticleMasses::k0sMSigma,
                                   k0sCollection) {
    setResMassRange(2.80, 3.40);
    setFlyingMassRange(0.00, 2.00);
    setMassRange(3.50, 8.00);
    setKinFitProbMin(0.02);
    setMassFitRange(5.00, 6.00);
    setConstr(true);
  }

  // deleted copy constructor and assignment operator
  BPHBdToJPsiKsBuilder(const BPHBdToJPsiKsBuilder& x) = delete;
  BPHBdToJPsiKsBuilder& operator=(const BPHBdToJPsiKsBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBdToJPsiKsBuilder() override {}

  /** Operations
   */
  /// set cuts
  void setJPsiMassMin(double m) { setResMassMin(m); }
  void setJPsiMassMax(double m) { setResMassMax(m); }
  void setK0MassMin(double m) { setFlyingMassMin(m); }
  void setK0MassMax(double m) { setFlyingMassMax(m); }

  /// get current cuts
  double getJPsiMassMin() const { return getResMassMin(); }
  double getJPsiMassMax() const { return getResMassMax(); }
  double getK0MassMin() const { return getFlyingMassMin(); }
  double getK0MassMax() const { return getFlyingMassMax(); }
};

#endif
