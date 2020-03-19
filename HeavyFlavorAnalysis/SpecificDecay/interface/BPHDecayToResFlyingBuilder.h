#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResFlyingBuilder_h
/** \class BPHDecayToResFlyingBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a resonances and a flying particle,
 *     both decaying to an opposite charged particles pair
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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKinFitChi2Select.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToResFlyingBuilder : public BPHDecayConstrainedBuilder {
public:
  /** Constructor
   */
  BPHDecayToResFlyingBuilder(const edm::EventSetup& es,
                             const std::string& resName,
                             double resMass,
                             double resWidth,
                             const std::vector<BPHPlusMinusConstCandPtr>& resCollection,
                             const std::string& flyName,
                             double flyMass,
                             double flyMSigma,
                             const std::vector<BPHPlusMinusConstCandPtr>& flyCollection);

  // deleted copy constructor and assignment operator
  BPHDecayToResFlyingBuilder(const BPHDecayToResFlyingBuilder& x) = delete;
  BPHDecayToResFlyingBuilder& operator=(const BPHDecayToResFlyingBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResFlyingBuilder() override;

  /** Operations
   */
  /// build candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// get original daughters map
  const std::map<const BPHRecoCandidate*, const BPHRecoCandidate*>& daughMap() const { return dMap; }

  /// set cuts
  void setFlyingMassMin(double m);
  void setFlyingMassMax(double m);
  void setFlyingMassRange(double mMin, double mMax);
  void setKinFitProbMin(double p);

  /// get current cuts
  double getFlyingMassMin() const { return flySel->getMassMin(); }
  double getFlyingMassMax() const { return flySel->getMassMax(); }
  double getKinFitProbMin() const { return kfChi2Sel->getProbMin(); }

private:
  std::string fName;
  double fMass;
  double fMSigma;

  const std::vector<BPHPlusMinusConstCandPtr>* fCollection;

  BPHMassFitSelect* flySel;
  BPHKinFitChi2Select* kfChi2Sel;

  std::map<const BPHRecoCandidate*, const BPHRecoCandidate*> dMap;
  std::vector<BPHRecoConstCandPtr> recList;
};

#endif
