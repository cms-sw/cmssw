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

class BPHBsToJPsiPhiBuilder : public BPHDecayToResResBuilder {
public:
  /** Constructor
   */
  BPHBsToJPsiPhiBuilder(const edm::EventSetup& es,
                        const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                        const std::vector<BPHPlusMinusConstCandPtr>& phiCollection)
      : BPHDecayToResResBuilder(es,
                                "JPsi",
                                BPHParticleMasses::jPsiMass,
                                BPHParticleMasses::jPsiMWidth,
                                jpsiCollection,
                                "Phi",
                                phiCollection) {
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
  ~BPHBsToJPsiPhiBuilder() override {}

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
};

#endif
