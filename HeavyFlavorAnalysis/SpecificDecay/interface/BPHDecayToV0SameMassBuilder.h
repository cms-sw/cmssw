#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0SameMassBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0SameMassBuilder_h
/** \class BPHDecayToV0SameMassBuilder
 *
 *  Description: 
 *     Class to build neutral particles decaying to a V0,
 *     with daughters having same mass,
 *     starting from reco::Candidates or already reconstructed V0s
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0Builder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToChargedXXbarBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToV0SameMassBuilder : public BPHDecayToV0Builder, public BPHDecayToChargedXXbarBuilder {
public:
  /** Constructor
   */
  BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& posName,
                              const std::string& negName,
                              double daugMass,
                              double daugSigma,
                              const BPHRecoBuilder::BPHGenericCollection* posCollection,
                              const BPHRecoBuilder::BPHGenericCollection* negCollection);
  BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& posName,
                              const std::string& negName,
                              double daugMass,
                              double daugSigma,
                              const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                              const std::string& searchList = "cfp");
  BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& posName,
                              const std::string& negName,
                              double daugMass,
                              double daugSigma,
                              const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                              const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHDecayToV0SameMassBuilder(const BPHDecayToV0SameMassBuilder& x) = delete;
  BPHDecayToV0SameMassBuilder& operator=(const BPHDecayToV0SameMassBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToV0SameMassBuilder() override = default;

protected:
  double pMass;
  double pSigma;

  /// build candidates and link to V0
  void buildFromBPHGenericCollection() override;
  BPHPlusMinusCandidatePtr buildCandidate(const reco::Candidate* c1,
                                          const reco::Candidate* c2,
                                          const void* v0,
                                          v0Type type) override;

private:
  /// build candidates
  void fillRecList() override { BPHDecayToV0Builder::fillRecList(); }
};

#endif
