#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHKx0ToKPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHKx0ToKPiBuilder_h
/** \class BPHKx0ToKPiBuilder
 *
 *  Description: 
 *     Class to build K*0 to K+ pi- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToTkpTknSymChargeBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
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

class BPHKx0ToKPiBuilder : public BPHDecayToTkpTknSymChargeBuilder {
public:
  /** Constructor
   */
  BPHKx0ToKPiBuilder(const BPHEventSetupWrapper& es,
                     const BPHRecoBuilder::BPHGenericCollection* posCollection,
                     const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayGenericBuilderBase(es),
        BPHDecayToTkpTknSymChargeBuilder(es,
                                         "Kaon",
                                         BPHParticleMasses::kaonMass,
                                         BPHParticleMasses::kaonMSigma,
                                         "Pion",
                                         BPHParticleMasses::pionMass,
                                         BPHParticleMasses::pionMSigma,
                                         posCollection,
                                         negCollection,
                                         BPHParticleMasses::kx0Mass) {
    setTrk1PtMin(0.7);
    setTrk2PtMin(0.7);
    setTrk1EtaMax(10.0);
    setTrk2EtaMax(10.0);
    setMassRange(0.75, 1.05);
    setProbMin(0.0);
  }

  // deleted copy constructor and assignment operator
  BPHKx0ToKPiBuilder(const BPHKx0ToKPiBuilder& x) = delete;
  BPHKx0ToKPiBuilder& operator=(const BPHKx0ToKPiBuilder& x) = delete;

  /** Destructor
   */
  ~BPHKx0ToKPiBuilder() override = default;

  /** Operations
   */
  /// set cuts
  void setPiPtMin(double pt) { setTrk1PtMin(pt); }
  void setPiEtaMax(double eta) { setTrk1EtaMax(eta); }
  void setKPtMin(double pt) { setTrk2PtMin(pt); }
  void setKEtaMax(double eta) { setTrk2EtaMax(eta); }
  void setPtMin(double pt) {
    setTrk1PtMin(pt);
    setTrk2PtMin(pt);
  }
  void setEtaMax(double eta) {
    setTrk1EtaMax(eta);
    setTrk2EtaMax(eta);
  }

  /// get current cuts
  double getPiPtMin() const { return getTrk1PtMin(); }
  double getPiEtaMax() const { return getTrk1EtaMax(); }
  double getKPtMin() const { return getTrk2PtMin(); }
  double getKEtaMax() const { return getTrk2EtaMax(); }
};

#endif
