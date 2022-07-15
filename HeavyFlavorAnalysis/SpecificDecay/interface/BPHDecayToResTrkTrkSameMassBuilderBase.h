#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkTrkSameMassBuilderBase_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkTrkSameMassBuilderBase_h
/** \class BPHDecayToResTrkTrkSameMassBuilderBase
 *
 *  Description: 
 *     Base class to build a particle decaying to a particle, decaying itself
 *     in cascade, and two additional opposite charged particles pair
 *     having the same mass
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

class BPHEventSetupWrapper;
class BPHParticleNeutralVeto;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToResTrkTrkSameMassBuilderBase : public virtual BPHDecayConstrainedBuilderBase {
public:
  /** Constructor
   */
  BPHDecayToResTrkTrkSameMassBuilderBase(const BPHEventSetupWrapper& es,
                                         const std::string& resName,
                                         double resMass,
                                         double resWidth,
                                         const std::string& posName,
                                         const std::string& negName,
                                         double trkMass,
                                         double trkSigma,
                                         const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                         const BPHRecoBuilder::BPHGenericCollection* negCollection);

  // deleted copy constructor and assignment operator
  BPHDecayToResTrkTrkSameMassBuilderBase(const BPHDecayToResTrkTrkSameMassBuilderBase& x) = delete;
  BPHDecayToResTrkTrkSameMassBuilderBase& operator=(const BPHDecayToResTrkTrkSameMassBuilderBase& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResTrkTrkSameMassBuilderBase() override = default;

  /** Operations
   */

  /// set cuts
  void setTrkPtMin(double pt);
  void setTrkEtaMax(double eta);

  /// get current cuts
  double getTrkPtMin() const { return ptMin; }
  double getTrkEtaMax() const { return etaMax; }

protected:
  BPHDecayToResTrkTrkSameMassBuilderBase(const std::string& posName,
                                         const std::string& negName,
                                         double trkMass,
                                         double trkSigma,
                                         const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                         const BPHRecoBuilder::BPHGenericCollection* negCollection);

  std::string pName;
  std::string nName;
  double tMass;
  double tSigma;

  const BPHRecoBuilder::BPHGenericCollection* pCollection;
  const BPHRecoBuilder::BPHGenericCollection* nCollection;

  double ptMin;
  double etaMax;

  void fillTrkTrkList();
  std::vector<BPHPlusMinusConstCandPtr> ttPairs;
};

#endif
