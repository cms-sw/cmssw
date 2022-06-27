#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToJPsiPiPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToJPsiPiPiBuilder_h
/** \class BPHDecayToJPsiPiPiBuilder
 *
 *  Description: 
 *     Class to build particles decaying to JPsi pi+ pi- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResTrkTrkSameMassBuilder.h"

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

class BPHDecayToJPsiPiPiBuilder : public BPHDecayToResTrkTrkSameMassBuilder<BPHRecoCandidate, BPHPlusMinusCandidate> {
public:
  /** Constructor
   */
  BPHDecayToJPsiPiPiBuilder(const BPHEventSetupWrapper& es,
                            const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                            const BPHRecoBuilder::BPHGenericCollection* posCollection,
                            const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase("JPsi", BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth),
        BPHDecayToResTrkTrkSameMassBuilder(jpsiCollection,
                                           "PionPos",
                                           "PionNeg",
                                           BPHParticleMasses::pionMass,
                                           BPHParticleMasses::pionMSigma,
                                           posCollection,
                                           negCollection) {
    setResMassRange(2.80, 3.40);
    setTrkPtMin(1.0);
    setTrkEtaMax(10.0);
    setMassRange(3.00, 4.50);
    setProbMin(0.02);
    setMassFitRange(3.50, 4.20);
  }

  // deleted copy constructor and assignment operator
  BPHDecayToJPsiPiPiBuilder(const BPHDecayToJPsiPiPiBuilder& x) = delete;
  BPHDecayToJPsiPiPiBuilder& operator=(const BPHDecayToJPsiPiPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToJPsiPiPiBuilder() override = default;

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

protected:
  BPHDecayToJPsiPiPiBuilder(const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                            const BPHRecoBuilder::BPHGenericCollection* posCollection,
                            const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayToResTrkTrkSameMassBuilder(jpsiCollection,
                                           "PionPos",
                                           "PionNeg",
                                           BPHParticleMasses::pionMass,
                                           BPHParticleMasses::pionMSigma,
                                           posCollection,
                                           negCollection) {
    rName = "JPsi";
    rMass = BPHParticleMasses::jPsiMass;
    rWidth = BPHParticleMasses::jPsiMWidth;
    resoSel = new BPHMassSelect(-2.0e+06, -1.0e+06);
    massConstr = true;
    mFitSel = new BPHMassFitSelect(rName, rMass, rWidth, -2.0e+06, -1.0e+06);
  }
};

#endif
