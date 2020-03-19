#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBuToJPsiKBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBuToJPsiKBuilder_h
/** \class BPHBuToJPsiKBuilder
 *
 *  Description: 
 *     Class to build B+- to JPsi K+- candidates
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

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHBuToJPsiKBuilder : public BPHDecayToResTrkBuilder {
public:
  /** Constructor
   */
  BPHBuToJPsiKBuilder(const edm::EventSetup& es,
                      const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                      const BPHRecoBuilder::BPHGenericCollection* kaonCollection)
      : BPHDecayToResTrkBuilder(es,
                                "JPsi",
                                BPHParticleMasses::jPsiMass,
                                BPHParticleMasses::jPsiMWidth,
                                jpsiCollection,
                                "Kaon",
                                BPHParticleMasses::kaonMass,
                                BPHParticleMasses::kaonMSigma,
                                kaonCollection) {
    setResMassRange(2.80, 3.40);
    setTrkPtMin(0.7);
    setTrkEtaMax(10.0);
    setMassRange(3.50, 8.00);
    setProbMin(0.02);
    setMassFitRange(5.00, 6.00);
    setConstr(true);
  }

  // deleted copy constructor and assignment operator
  BPHBuToJPsiKBuilder(const BPHBuToJPsiKBuilder& x) = delete;
  BPHBuToJPsiKBuilder& operator=(const BPHBuToJPsiKBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBuToJPsiKBuilder() override {}

  /** Operations
   */
  /// set cuts
  void setKPtMin(double pt) { setTrkPtMin(pt); }
  void setKEtaMax(double eta) { setTrkEtaMax(eta); }
  void setJPsiMassMin(double m) { setResMassMin(m); }
  void setJPsiMassMax(double m) { setResMassMax(m); }

  /// get current cuts
  double getKPtMin() const { return getTrkPtMin(); }
  double getKEtaMax() const { return getTrkEtaMax(); }
  double getJPsiMassMin() const { return getResMassMin(); }
  double getJPsiMassMax() const { return getResMassMax(); }
};

#endif
