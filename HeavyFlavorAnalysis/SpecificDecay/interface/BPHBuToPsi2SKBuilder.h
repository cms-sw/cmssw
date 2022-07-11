#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBuToPsi2SKBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBuToPsi2SKBuilder_h
/** \class BPHBuToPsi2SKBuilder
 *
 *  Description: 
 *     Class to build B+- to Psi2S K+- candidates
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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/MultiTrackMassKinematicConstraint.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <iostream>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHBuToPsi2SKBuilder : public BPHDecayToResTrkBuilder<BPHRecoCandidate, BPHRecoCandidate> {
public:
  /** Constructor
   */
  BPHBuToPsi2SKBuilder(const BPHEventSetupWrapper& es,
                       const std::vector<BPHRecoConstCandPtr>& psi2SCollection,
                       const BPHRecoBuilder::BPHGenericCollection* kaonCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase("Psi2S", BPHParticleMasses::psi2Mass, BPHParticleMasses::psi2MWidth),
        BPHDecayToResTrkBuilder(
            psi2SCollection, "Kaon", BPHParticleMasses::kaonMass, BPHParticleMasses::kaonMSigma, kaonCollection) {
    setResMassRange(3.30, 4.00);
    setTrkPtMin(0.7);
    setTrkEtaMax(10.0);
    setMassRange(3.50, 8.00);
    setProbMin(0.02);
  }

  // deleted copy constructor and assignment operator
  BPHBuToPsi2SKBuilder(const BPHBuToPsi2SKBuilder& x) = delete;
  BPHBuToPsi2SKBuilder& operator=(const BPHBuToPsi2SKBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBuToPsi2SKBuilder() override = default;

  /** Operations
   */
  /// get original daughters map
  const std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>& daughMap() const { return dMap; }

  /// set cuts
  void setKPtMin(double pt) { setTrkPtMin(pt); }
  void setKEtaMax(double eta) { setTrkEtaMax(eta); }
  void setPsi2SMassMin(double m) { setResMassMin(m); }
  void setPsi2SMassMax(double m) { setResMassMax(m); }

  /// get current cuts
  double getKPtMin() const { return getTrkPtMin(); }
  double getKEtaMax() const { return getTrkEtaMax(); }
  double getPsi2SMassMin() const { return getResMassMin(); }
  double getPsi2SMassMax() const { return getResMassMax(); }

  /// setup parameters for BPHRecoBuilder
  void setup(void* parameters) override {}

private:
  std::map<const BPHRecoCandidate*, const BPHRecoCandidate*> dMap;
};

#endif
