#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0DiffMassBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0DiffMassBuilder_h
/** \class BPHDecayToV0DiffMassBuilder
 *
 *  Description: 
 *     Class to build neutral particles decaying to a V0,
 *     with daughters having different mass,
 *     starting from reco::Candidates or already reconstructed V0s
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0Builder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToV0DiffMassBuilder : public BPHDecayToV0Builder {
public:
  /** Constructor
   */
  BPHDecayToV0DiffMassBuilder(const edm::EventSetup& es,
                              const std::string& d1Name,
                              double d1Mass,
                              double d1Sigma,
                              const std::string& d2Name,
                              double d2Mass,
                              double d2Sigma,
                              const BPHRecoBuilder::BPHGenericCollection* posCollection,
                              const BPHRecoBuilder::BPHGenericCollection* negCollection,
                              double expectedMass);
  BPHDecayToV0DiffMassBuilder(const edm::EventSetup& es,
                              const std::string& d1Name,
                              double d1Mass,
                              double d1Sigma,
                              const std::string& d2Name,
                              double d2Mass,
                              double d2Sigma,
                              const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                              double expectedMass,
                              const std::string& searchList = "cfp");
  BPHDecayToV0DiffMassBuilder(const edm::EventSetup& es,
                              const std::string& d1Name,
                              double d1Mass,
                              double d1Sigma,
                              const std::string& d2Name,
                              double d2Mass,
                              double d2Sigma,
                              const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                              double expectedMass,
                              const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHDecayToV0DiffMassBuilder(const BPHDecayToV0DiffMassBuilder& x) = delete;
  BPHDecayToV0DiffMassBuilder& operator=(const BPHDecayToV0DiffMassBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToV0DiffMassBuilder() override;

protected:
  double p1Mass;
  double p2Mass;
  double p1Sigma;
  double p2Sigma;
  double expMass;

  /// build candidates and link to V0
  void buildFromBPHGenericCollection() override;
  BPHPlusMinusCandidatePtr buildCandidate(const reco::Candidate* c1,
                                          const reco::Candidate* c2,
                                          const void* v0,
                                          v0Type type) override;
};

#endif
