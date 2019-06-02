#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleNeutralVeto_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleNeutralVeto_h
/** \class BPHParticleNeutralVeto
 *
 *  Description: 
 *     Class for neutral particle rejection
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHParticleNeutralVeto : public BPHRecoSelect {
public:
  /** Constructor
   */
  BPHParticleNeutralVeto() {}

  /** Destructor
   */
  ~BPHParticleNeutralVeto() override {}

  /** Operations
   */
  /// select charged particles
  bool accept(const reco::Candidate& cand) const override { return (cand.charge() != 0); }

private:
  // private copy and assigment constructors
  BPHParticleNeutralVeto(const BPHParticleNeutralVeto& x) = delete;
  BPHParticleNeutralVeto& operator=(const BPHParticleNeutralVeto& x) = delete;
};

#endif
