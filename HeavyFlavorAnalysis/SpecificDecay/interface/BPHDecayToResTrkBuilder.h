#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilder_h
/** \class BPHDecayToResTrkBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a resonance, decaying itself
 *     to an opposite charged particles pair, and an additional track
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

class BPHParticleNeutralVeto;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToResTrkBuilder : public BPHDecayConstrainedBuilder {
public:
  /** Constructor
   */
  BPHDecayToResTrkBuilder(const edm::EventSetup& es,
                          const std::string& resName,
                          double resMass,
                          double resWidth,
                          const std::vector<BPHPlusMinusConstCandPtr>& resCollection,
                          const std::string& trkName,
                          double trkMass,
                          double trkSigma,
                          const BPHRecoBuilder::BPHGenericCollection* trkCollection);

  // deleted copy constructor and assignment operator
  BPHDecayToResTrkBuilder(const BPHDecayToResTrkBuilder& x) = delete;
  BPHDecayToResTrkBuilder& operator=(const BPHDecayToResTrkBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResTrkBuilder() override;

  /** Operations
   */
  /// build candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setTrkPtMin(double pt);
  void setTrkEtaMax(double eta);

  /// get current cuts
  double getTrkPtMin() const { return ptSel->getPtMin(); }
  double getTrkEtaMax() const { return etaSel->getEtaMax(); }

private:
  std::string tName;
  double tMass;
  double tSigma;

  const BPHRecoBuilder::BPHGenericCollection* tCollection;

  BPHParticleNeutralVeto* tknVeto;
  BPHParticlePtSelect* ptSel;
  BPHParticleEtaSelect* etaSel;

  std::vector<BPHRecoConstCandPtr> recList;
};

#endif
