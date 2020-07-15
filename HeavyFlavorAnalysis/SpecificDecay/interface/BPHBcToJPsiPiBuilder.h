#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBcToJPsiPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBcToJPsiPiBuilder_h
/** \class BPHBcToJPsiPiBuilder
 *
 *  Description: 
 *     Class to build Bc to JPsi pi+- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResTrkBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

#include "FWCore/Framework/interface/Event.h"

class BPHParticleNeutralVeto;
class BPHParticlePtSelect;
class BPHParticleEtaSelect;
class BPHMassSelect;
class BPHChi2Select;
class BPHMassFitSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHBcToJPsiPiBuilder : public BPHDecayToResTrkBuilder {
public:
  /** Constructor
   */
  BPHBcToJPsiPiBuilder(const edm::EventSetup& es,
                       const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                       const BPHRecoBuilder::BPHGenericCollection* pionCollection)
      : BPHDecayToResTrkBuilder(es,
                                "JPsi",
                                BPHParticleMasses::jPsiMass,
                                BPHParticleMasses::jPsiMWidth,
                                jpsiCollection,
                                "Pion",
                                BPHParticleMasses::pionMass,
                                BPHParticleMasses::pionMSigma,
                                pionCollection) {
    setResMassRange(2.80, 3.40);
    setTrkPtMin(0.7);
    setTrkEtaMax(10.0);
    setMassRange(4.00, 9.00);
    setProbMin(0.02);
    setMassFitRange(6.00, 7.00);
    setConstr(true);
  }

  // deleted copy constructor and assignment operator
  BPHBcToJPsiPiBuilder(const BPHBcToJPsiPiBuilder& x) = delete;
  BPHBcToJPsiPiBuilder& operator=(const BPHBcToJPsiPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBcToJPsiPiBuilder() override {}

  /** Operations
   */
  /// set cuts
  void setPiPtMin(double pt) { setTrkPtMin(pt); }
  void setPiEtaMax(double eta) { setTrkEtaMax(eta); }
  void setJPsiMassMin(double m) { setResMassMin(m); }
  void setJPsiMassMax(double m) { setResMassMax(m); }

  /// get current cuts
  double getPiPtMin() const { return getTrkPtMin(); }
  double getPiEtaMax() const { return getTrkEtaMax(); }
  double getJPsiMassMin() const { return getResMassMin(); }
  double getJPsiMassMax() const { return getResMassMax(); }
};

#endif
