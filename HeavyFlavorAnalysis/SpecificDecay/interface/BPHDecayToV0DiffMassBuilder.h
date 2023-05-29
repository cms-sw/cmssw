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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToTkpTknSymChargeBuilder.h"

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

class BPHDecayToV0DiffMassBuilder : public BPHDecayToV0Builder, public BPHDecayToTkpTknSymChargeBuilder {
public:
  /** Constructor
   */
  BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& daug1Name,
                              double daug1Mass,
                              double daug1Sigma,
                              const std::string& daug2Name,
                              double daug2Mass,
                              double daug2Sigma,
                              const BPHRecoBuilder::BPHGenericCollection* posCollection,
                              const BPHRecoBuilder::BPHGenericCollection* negCollection,
                              double expectedMass);
  BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& daug1Name,
                              double daug1Mass,
                              double daug1Sigma,
                              const std::string& daug2Name,
                              double daug2Mass,
                              double daug2Sigma,
                              const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                              double expectedMass,
                              const std::string& searchList = "cfp");
  BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                              const std::string& daug1Name,
                              double daug1Mass,
                              double daug1Sigma,
                              const std::string& daug2Name,
                              double daug2Mass,
                              double daug2Sigma,
                              const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                              double expectedMass,
                              const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHDecayToV0DiffMassBuilder(const BPHDecayToV0DiffMassBuilder& x) = delete;
  BPHDecayToV0DiffMassBuilder& operator=(const BPHDecayToV0DiffMassBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToV0DiffMassBuilder() override = default;

  /** Operations
   */

  /// set cuts
  void setPtMin(double pt) {
    setTrk1PtMin(pt);
    setTrk2PtMin(pt);
  }
  void setEtaMax(double eta) {
    setTrk1EtaMax(eta);
    setTrk2EtaMax(eta);
  }

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

private:
  /// build candidates
  void fillRecList() override { BPHDecayToV0Builder::fillRecList(); }
};

#endif
